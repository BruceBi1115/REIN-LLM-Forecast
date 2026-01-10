# trainer.py (REGRESSION version - full file)  [RESIDUAL BASE+DELTA, NO TRUE-vs-SHUFFLED]

from __future__ import annotations

import csv
import gc
import os
import json
import math
from collections import deque

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

from src.data_construction.DataStatistic import DataStatistic
from .utils.utils import set_seed, device_from_id
from .data_construction.data import make_loader
from .news_rules import load_news, get_candidates, select_news, _load_keywords
from .data_construction.prompt import format_news, load_templates, build_prompt
from .RL.rl_bandit import LinTS, LinUCB, RewardNormalizer
from .ValidationState import ValidationState
from .utils.logger import setup_live_logger
from .RL.features import bandit_select, get_context_features, encode_instruction

from .model import load_checkpoint, load_llama_lora, save_checkpoint
from .residual_utils import freeze_module, zero_regressor_head, split_two_stage_epochs

dataStatistic = DataStatistic()


def _zstats(x, eps: float = 1e-6):
    x = np.asarray(x, dtype=np.float32)
    mu = float(x.mean())
    sigma = float(x.std())
    if sigma < eps:
        sigma = eps
    return mu, sigma


def _zscore(x, mu, sigma):
    x = np.asarray(x, dtype=np.float32)
    return ((x - mu) / sigma).tolist()


def _inv_zscore(z, mu, sigma):
    z = np.asarray(z, dtype=np.float32)
    return (z * sigma + mu).tolist()


def _maybe_news_dropout(news_str: str, args) -> str:
    p = float(getattr(args, "news_dropout", 0.0) or 0.0)
    if p <= 0:
        return news_str
    if np.random.rand() < p:
        return ""
    return news_str


def _make_patches(seq: list[float], patch_len: int, stride: int):
    """
    seq: length L list
    returns: patches (P, patch_len), mask (P,)
    """
    x = np.asarray(seq, dtype=np.float32)
    L = int(x.shape[0])
    patch_len = int(patch_len)
    stride = int(stride)

    if patch_len <= 0:
        raise ValueError("patch_len must be > 0")
    if stride <= 0:
        raise ValueError("patch_stride must be > 0")

    if L < patch_len:
        p = np.zeros((1, patch_len), dtype=np.float32)
        p[0, :L] = x
        m = np.ones((1,), dtype=np.int64)
        return p, m

    idxs = list(range(0, L - patch_len + 1, stride))
    patches = np.stack([x[i: i + patch_len] for i in idxs], axis=0).astype(np.float32)  # (P, patch_len)
    mask = np.ones((patches.shape[0],), dtype=np.int64)
    return patches, mask


def history_text(history_z: list[float], mu: float, sigma: float) -> str:
    hz = np.asarray(history_z, dtype=np.float32)
    last = hz.tolist() if len(hz) >= 8 else hz.tolist()
    slope = float(hz[-1] - hz[-9]) if len(hz) >= 9 else float(hz[-1] - hz[0]) if len(hz) >= 2 else 0.0
    return (
        f"History z-scored (mu={mu:.4f}, std={sigma:.4f}). "
        f"Last {len(last)} z: {', '.join([f'{v:.3f}' for v in last])}. "
        f"Recent slope: {slope:.3f}."
    )


def _pad_2d_int(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(seqs)
    max_t = max(len(s) for s in seqs) if B > 0 else 1
    input_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_t), dtype=torch.long)
    for i, s in enumerate(seqs):
        t = len(s)
        if t == 0:
            continue
        input_ids[i, :t] = torch.tensor(s, dtype=torch.long)
        attn[i, :t] = 1
    return input_ids, attn


def _pad_patches(patches_list: list[np.ndarray], mask_list: list[np.ndarray], patch_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    B = len(patches_list)
    max_p = max(p.shape[0] for p in patches_list) if B > 0 else 1
    ts_patches = torch.zeros((B, max_p, patch_len), dtype=torch.float32)
    ts_patch_mask = torch.zeros((B, max_p), dtype=torch.long)
    for i, (p, pm) in enumerate(zip(patches_list, mask_list)):
        P_i = p.shape[0]
        ts_patches[i, :P_i, :] = torch.tensor(p, dtype=torch.float32)
        ts_patch_mask[i, :P_i] = torch.tensor(pm, dtype=torch.long)
    return ts_patches, ts_patch_mask


def build_batch_inputs(
    batch,
    tokenizer,
    templates,
    tpl_id,
    args,
    news_df,
    policy_name,
    policy_kw,
    volatility_bin,
    epoch: int = -1,
    record_train_prompt: bool = False,
    testing: bool = False,
    force_no_news: bool = False,
    news_dropout: bool = False,
):
    """
    Returns:
      input_ids, attn,
      ts_patches, ts_patch_mask,
      targets_z, metas,
      prompt_texts
    """
    L, H = int(args.history_len), int(args.horizon)
    news_budget = int(args.token_budget * args.token_budget_news_frac)

    patch_len = int(getattr(args, "patch_len", 4))
    patch_stride = int(getattr(args, "patch_stride", patch_len))

    tpl_text = templates[tpl_id]["text"]
    B = len(batch["history_value"])

    targets_z_list = []
    patches_list = []
    patchmask_list = []
    metas = []

    hist_strs = []
    news_str_list = []

    start_dates = []
    end_dates = []
    pred_starts = []
    pred_ends = []

    len_selected_news = []

    for i in range(B):
        history = batch["history_value"][i].tolist()
        target = batch["target_value"][i].tolist()
        t_target = batch["target_time"][i]

        mu, sigma = _zstats(history, eps=float(getattr(args, "zscore_eps", 1e-6)))
        history_z = _zscore(history, mu, sigma)
        target_z = _zscore(target, mu, sigma)

        p, pm = _make_patches(history_z, patch_len=patch_len, stride=patch_stride)

        # news
        if force_no_news or (news_df is None) or (len(news_df) == 0) or (policy_name == "none"):
            selected = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])
        else:
            cand = get_candidates(news_df, args.news_time_col, t_target, args.news_window_days, args.news_topM)
            selected = select_news(cand, policy_name, args.news_text_col, policy_kw, args.news_topK)

        len_selected_news.append(len(selected))

        news_str = ""
        if not force_no_news and len(selected) > 0:
            news_str = format_news(
                selected,
                args.news_text_col,
                news_budget,
                tokenizer,
                summary_method=args.news_summary_method,
                max_sentences=args.news_max_sentences,
            )
            if news_dropout:
                news_str = _maybe_news_dropout(news_str, args)

        # time meta for prompt
        start_date = batch["history_times"][0][i]
        end_date = batch["history_times"][-1][i]
        prediction_start = batch["target_times"][0][i]
        prediction_end = batch["target_times"][-1][i]

        targets_z_list.append(np.asarray(target_z, dtype=np.float32))
        patches_list.append(p)
        patchmask_list.append(pm)
        metas.append({"mu": mu, "sigma": sigma})

        hist_strs.append(history_text(history_z, mu, sigma))
        news_str_list.append(news_str)

        start_dates.append(start_date)
        end_dates.append(end_date)
        pred_starts.append(prediction_start)
        pred_ends.append(prediction_end)

    # tokenize prompts
    ids_list = []
    prompt_texts = []

    for i in range(B):
        prompt = build_prompt(
            tpl_text,
            L,
            H,
            args.unit,
            args.description,
            hist_strs[i],
            news_str_list[i],
            start_date=start_dates[i],
            end_date=end_dates[i],
            freq=args.freq_min,
            value_col=args.value_col,
            pred_end=pred_ends[i],
            pred_start=pred_starts[i],
            region=args.region,
        )
        prompt = prompt + "\n\n[Output]\n" + f"Predict the next {H} steps (internally as z-values).\n"

        # stats (only track once per call; you were tracking per prompt anyway)
        dataStatistic.news_num_stats_update(len_selected_news[i], prompt=prompt)

        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=int(args.max_seq_len),
            return_attention_mask=False,  # we build attn ourselves for speed
        )
        ids_list.append(enc["input_ids"])
        prompt_texts.append(prompt)

        if record_train_prompt:
            ckpt_dir = os.path.join("./checkpoints", args.taskName)
            os.makedirs(ckpt_dir, exist_ok=True)
            prompt_path = os.path.join(ckpt_dir, f"prompts_{args.taskName}.json")
            rec = {
                "batch_idx": i,
                "epoch_num": epoch + 1,
                "template_id": int(tpl_id),
                "policy": str(policy_name),
                "force_no_news": bool(force_no_news),
                "prompt": prompt,
                "mu": float(metas[i]["mu"]),
                "sigma": float(metas[i]["sigma"]),
            }
            with open(prompt_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    input_ids, attn = _pad_2d_int(ids_list, pad_id=tokenizer.pad_token_id)
    ts_patches, ts_patch_mask = _pad_patches(patches_list, patchmask_list, patch_len=patch_len)
    targets_z = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets_z_list], dim=0)

    return input_ids, attn, ts_patches, ts_patch_mask, targets_z, metas, prompt_texts


@torch.no_grad()
def evaluate_metrics_single(
    model,
    tokenizer,
    data_loader,
    templates,
    tpl_id,
    args,
    news_df,
    policy_name,
    policy_kw,
    device,
    volatility_bin,
    testing: bool = False,
    true_pred_csv_path: str | None = None,
    news_dropout: bool = False,
    force_no_news: bool = False,
):
    """
    Single-model evaluation: used for BASE stage.
    Returns:
      - loss_avg: z-space MSE
      - mse_avg: raw-scale MSE
      - mae_avg: raw-scale MAE
    """
    model.eval()

    loss_sum, n_samples = 0.0, 0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0

    if testing:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_json_path = os.path.join(ckpt_dir, f"test_answers_{args.taskName}.json")

    for _, batch in enumerate(data_loader):
        input_ids, attn, ts_patches, ts_patch_mask, targets_z, metas, prompt_texts = build_batch_inputs(
            batch=batch,
            tokenizer=tokenizer,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            volatility_bin=volatility_bin,
            epoch=-1,
            record_train_prompt=False,
            testing=testing,
            force_no_news=force_no_news,
            news_dropout=news_dropout,
        )

        input_ids = input_ids.to(device)
        attn = attn.to(device)
        ts_patches = ts_patches.to(device)
        ts_patch_mask = ts_patch_mask.to(device)
        targets_z = targets_z.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            ts_patches=ts_patches,
            ts_patch_mask=ts_patch_mask,
            targets=targets_z,
        )
        loss = out["loss"]
        pred_z = out["pred"]

        bs = input_ids.size(0)
        loss_sum += float(loss.detach().cpu()) * bs
        n_samples += bs

        pred_z_cpu = pred_z.detach().to(torch.float32).cpu().numpy()
        targets_cpu = batch["target_value"].detach().cpu().numpy()  # raw

        for i in range(bs):
            mu = float(metas[i]["mu"])
            sigma = float(metas[i]["sigma"])

            pred_denorm = _inv_zscore(pred_z_cpu[i].tolist(), mu, sigma)
            true_vals = targets_cpu[i].reshape(-1).tolist()
            true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

            pred = np.asarray(pred_denorm, dtype=np.float32)
            true = np.asarray(true_vals, dtype=np.float32)

            se_sum += float(((pred - true) ** 2).sum())
            ae_sum += float(np.abs(pred - true).sum())
            n_elems += int(args.horizon)

            if true_pred_csv_path is not None:
                with open(true_pred_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(pred_denorm, true_vals))

            if testing:
                record = {
                    "test_prompt": prompt_texts[i],
                    "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                    "pred": [float(x) for x in pred_denorm],
                    "true": [float(x) for x in true_vals],
                    "mu": mu,
                    "sigma": sigma,
                }
                with open(ans_json_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    return loss_avg, mse_avg, mae_avg


@torch.no_grad()
def evaluate_metrics_residual(
    base_model,
    delta_model,
    tokenizer,
    data_loader,
    templates,
    tpl_id,
    args,
    news_df,
    policy_name,
    policy_kw,
    device,
    volatility_bin,
    testing: bool = False,
    true_pred_csv_path: str | None = None,
    news_dropout: bool = False,
):
    """
    Residual evaluation: pred = base(no-news) + delta(with-news)
    Returns combined metrics.
    """
    base_model.eval()
    delta_model.eval()

    loss_sum, n_samples = 0.0, 0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0

    if testing:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_json_path = os.path.join(ckpt_dir, f"test_answers_{args.taskName}.json")

    for _, batch in enumerate(data_loader):
        # build delta (with news)
        ids_d, attn_d, ts_p, ts_pm, targets_z, metas, prompt_texts = build_batch_inputs(
            batch=batch,
            tokenizer=tokenizer,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            volatility_bin=volatility_bin,
            epoch=-1,
            record_train_prompt=False,
            testing=testing,
            force_no_news=False,
            news_dropout=news_dropout,
        )
        # build base (no news) -- reuse patches/targets, only rebuild text ids
        ids_b, attn_b, _, _, _, _, _ = build_batch_inputs(
            batch=batch,
            tokenizer=tokenizer,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name="none",
            policy_kw=policy_kw,
            volatility_bin=volatility_bin,
            epoch=-1,
            record_train_prompt=False,
            testing=False,
            force_no_news=True,
            news_dropout=False,
        )

        ids_d = ids_d.to(device)
        attn_d = attn_d.to(device)
        ids_b = ids_b.to(device)
        attn_b = attn_b.to(device)

        ts_p = ts_p.to(device)
        ts_pm = ts_pm.to(device)
        targets_z = targets_z.to(device)

        out_base = base_model(
            input_ids=ids_b,
            attention_mask=attn_b,
            ts_patches=ts_p,
            ts_patch_mask=ts_pm,
            targets=None,
        )
        base_pred = out_base["pred"]

        out_delta = delta_model(
            input_ids=ids_d,
            attention_mask=attn_d,
            ts_patches=ts_p,
            ts_patch_mask=ts_pm,
            targets=None,
        )
        delta_pred = out_delta["pred"]

        pred_z = base_pred + delta_pred

        # z-space loss (combined objective)
        loss = F.mse_loss(pred_z.to(torch.float32), targets_z.to(torch.float32), reduction="mean")

        bs = ids_d.size(0)
        loss_sum += float(loss.detach().cpu()) * bs
        n_samples += bs

        pred_z_cpu = pred_z.detach().to(torch.float32).cpu().numpy()
        targets_cpu = batch["target_value"].detach().cpu().numpy()  # raw

        for i in range(bs):
            mu = float(metas[i]["mu"])
            sigma = float(metas[i]["sigma"])

            pred_denorm = _inv_zscore(pred_z_cpu[i].tolist(), mu, sigma)
            true_vals = targets_cpu[i].reshape(-1).tolist()
            true_vals = [float(x) for x in true_vals[: int(args.horizon)]]

            pred = np.asarray(pred_denorm, dtype=np.float32)
            true = np.asarray(true_vals, dtype=np.float32)

            se_sum += float(((pred - true) ** 2).sum())
            ae_sum += float(np.abs(pred - true).sum())
            n_elems += int(args.horizon)

            if true_pred_csv_path is not None:
                with open(true_pred_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(pred_denorm, true_vals))

            if testing:
                record = {
                    "test_prompt": prompt_texts[i],
                    "pred_z": [float(x) for x in pred_z_cpu[i].tolist()],
                    "pred": [float(x) for x in pred_denorm],
                    "true": [float(x) for x in true_vals],
                    "mu": mu,
                    "sigma": sigma,
                    "policy": str(policy_name),
                    "template_id": int(tpl_id),
                }
                with open(ans_json_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg = se_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    mae_avg = ae_sum / max(1, n_elems) if n_elems > 0 else float("inf")
    return loss_avg, mse_avg, mae_avg


def compute_volatility_bin(df, time_col="", value_col="", window=48, bins=10, dayfirst=True):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=dayfirst)
    df = df.sort_values(time_col)

    recent = df[value_col].iloc[-window:]
    if len(recent) < 2:
        return 0

    vol = recent.std()
    all_std = df[value_col].rolling(window).std().dropna()
    if len(all_std) == 0:
        return 0

    thresholds = np.quantile(all_std, np.linspace(0, 1, bins + 1)[1:-1])
    bin_id = np.digitize(vol, thresholds, right=True)
    return int(min(bin_id, bins - 1))


def make_tpl_feature_fn(templates, add_one_hot=True, add_cost_proxy=False, add_cross_terms=False):
    if isinstance(templates, dict):
        tpl_by_id = templates
    else:
        tpl_by_id = {int(t["id"]): t for t in templates}

    tpl_ids = sorted(tpl_by_id.keys())
    T = len(tpl_ids)

    id2idx = {tid: i for i, tid in enumerate(tpl_ids)}
    I = np.eye(T, dtype=np.float32)

    tpl_list = [tpl_by_id[tid] for tid in tpl_ids]
    n_paths_list = [float(t.get("n_paths", 1) or 1) for t in tpl_list]
    max_n_paths = max(n_paths_list) if n_paths_list else 1.0

    raw_breath_intensity = []
    for t in tpl_list:
        hb = float(bool(t.get("has_breath", False)))
        bf = float(t.get("breath_freq", 0) or 0)
        raw_breath_intensity.append(hb * (1.0 / bf) if hb > 0 and bf > 0 else 0.0)

    bi_min = min(raw_breath_intensity) if raw_breath_intensity else 0.0
    bi_max = max(raw_breath_intensity) if raw_breath_intensity else 1.0
    bi_range = (bi_max - bi_min) if bi_max > bi_min else 1.0

    def _cost_proxy(t):
        he = float(bool(t.get("has_example", False)))
        hb = float(bool(t.get("has_breath", False)))
        hd = float(bool(t.get("has_decomp", False)))
        hsc = float(bool(t.get("has_self_consistency", False)))
        np_norm = float(t.get("n_paths", 1) or 1) / max_n_paths
        return 0.4 * he + 0.5 * hd + 1.0 * hsc + 0.6 * np_norm + 0.2 * hb

    raw_costs = [_cost_proxy(t) for t in tpl_list]
    c_min = min(raw_costs) if raw_costs else 0.0
    c_max = max(raw_costs) if raw_costs else 1.0
    c_range = (c_max - c_min) if c_max > c_min else 1.0

    def _single_tpl_vec(tid: int) -> np.ndarray:
        t = tpl_by_id[int(tid)]

        he = float(bool(t.get("has_example", False)))
        hb = float(bool(t.get("has_breath", False)))
        hd = float(bool(t.get("has_decomp", False)))
        hsc = float(bool(t.get("has_self_consistency", False)))
        np_norm = float(t.get("n_paths", 1) or 1) / max_n_paths

        bf = float(t.get("breath_freq", 0) or 0)
        bi = hb * (1.0 / bf) if hb > 0 and bf > 0 else 0.0
        bi_norm = (bi - bi_min) / bi_range

        vec = [1.0, he, hb, hd, hsc, np_norm, bi_norm]

        if add_cost_proxy:
            vec.append((_cost_proxy(t) - c_min) / c_range)

        if add_one_hot:
            vec.extend(I[id2idx[tid]].tolist())

        return np.asarray(vec, dtype=np.float32)

    def tpl_features(tid: int, context_vector) -> np.ndarray:
        arm = _single_tpl_vec(tid)
        if add_cross_terms:
            if context_vector is None:
                raise ValueError("add_cross_terms=True requires context_vector")
            cross = np.outer(context_vector.astype(np.float32), arm).ravel()
            return np.concatenate([arm, cross]).astype(np.float32)
        return arm

    def feat_dim(context_dim) -> int:
        base = len(_single_tpl_vec(tpl_ids[0]))
        if add_cross_terms:
            return base + base * context_dim
        return base

    return tpl_features, feat_dim


def bandit_round_update_residual(
    base_model,
    delta_model,
    tokenizer,
    probe_loader,
    templates,
    allowed_tpl_ids,
    news_df,
    policy_space,
    policy_kw,
    args,
    device,
    volatility_bin,
    context_vector,
    tpl_features,
    bandit_tpl,
    bandit_pol,
    normalizer,
    live_logger,
    round_id,
    bidx,
    global_step,
):
    base_model.eval()
    delta_model.eval()

    cand = bandit_select(
        args,
        context_vector,
        live_logger,
        allowed_tpl_ids,
        policy_space,
        bandit_tpl,
        bandit_pol,
        tpl_features,
        pol_features=None,
        epoch=round_id,
        bidx=bidx,
        global_step=global_step,
    )
    tpl_id = cand["tpl_id"]
    policy_name = cand["policy_name"]
    pol_idx = cand["pol_idx"]

    probe_loss, probe_mse, probe_mae = evaluate_metrics_residual(
        base_model=base_model,
        delta_model=delta_model,
        tokenizer=tokenizer,
        data_loader=probe_loader,
        templates=templates,
        tpl_id=tpl_id,
        args=args,
        news_df=news_df,
        policy_name=policy_name,
        policy_kw=policy_kw,
        device=device,
        volatility_bin=volatility_bin,
        testing=False,
        true_pred_csv_path=None,
        news_dropout=False,
    )

    if args.reward_metric == "loss":
        metric_now = probe_loss
    elif args.reward_metric == "mse":
        metric_now = probe_mse
    else:
        metric_now = probe_mae

    r = -metric_now
    r_hat = normalizer.update_and_normalize(
        r, group_key=(args.region, args.horizon) if args.domain_reward_norm else None
    )

    x_tpl = np.concatenate([context_vector, tpl_features(tpl_id, context_vector)], axis=0).astype(np.float32)
    x_pol = context_vector.astype(np.float32)

    bandit_tpl.update(x_tpl, r_hat)
    bandit_pol.update(x_pol, r_hat)

    live_logger.info(
        f"BANDIT_ROUND(residual) round={round_id} tpl_id={tpl_id} policy={policy_name} "
        f"probe_loss={probe_loss:.6f} probe_mse={probe_mse:.6f} probe_mae={probe_mae:.6f} "
        f"reward_norm={r_hat:.6f}"
    )

    return tpl_id, policy_name, pol_idx


def main(args):
    filename = "log_residual_epoch_" + str(args.epochs) + "_" + args.taskName
    log_filename = filename + ".log"
    live_logger, live_path, log_jsonl = setup_live_logger(
        save_dir=args.save_dir + "/" + args.taskName, filename=log_filename
    )
    print(f"[live log] {live_path}  (实时查看: tail -f '{live_path}')")

    ckpt_dir = os.path.join("./checkpoints", args.taskName)
    os.makedirs(ckpt_dir, exist_ok=True)

    # clean outputs
    prompt_path = os.path.join(ckpt_dir, f"prompts_{args.taskName}.json")
    with open(prompt_path, "w", encoding="utf-8"):
        pass

    ans_json_path = os.path.join(ckpt_dir, f"test_answers_{args.taskName}.json")
    with open(ans_json_path, "w", encoding="utf-8"):
        pass

    true_pred_csv_path = os.path.join(ckpt_dir, f"true_pred_{args.taskName}.csv")
    with open(true_pred_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pred", "true"])

    set_seed(args.seed)
    device = device_from_id(args.gpu)

    def _read(path):
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    train_df = _read(args.train_file)
    val_df = _read(args.val_file)
    test_df = _read(args.test_file)

    train_df[args.time_col] = pd.to_datetime(train_df[args.time_col], dayfirst=args.dayFirst)
    val_df[args.time_col] = pd.to_datetime(val_df[args.time_col], dayfirst=args.dayFirst)
    test_df[args.time_col] = pd.to_datetime(test_df[args.time_col], dayfirst=args.dayFirst)

    train_loader = make_loader(
        train_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=True,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )
    val_loader = make_loader(
        val_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=False,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )
    test_loader = make_loader(
        test_df,
        args.time_col,
        args.value_col,
        args.history_len,
        args.horizon,
        args.stride,
        args.batch_size,
        shuffle=False,
        id_col=args.id_col,
        dayFirst=args.dayFirst,
    )

    # news
    news_df = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])
    news_df[args.news_time_col] = pd.to_datetime(news_df[args.news_time_col], dayfirst=args.dayFirst)
    if args.news_path:
        news_df = load_news(args.news_path, args.news_time_col, args.news_tz)

    policy_kw = _load_keywords(args.keyword_path)
    templates = load_templates(args.template_pool)

    patch_len = int(getattr(args, "patch_len", 4))

    volatility_bin = compute_volatility_bin(
        train_df,
        time_col=args.time_col,
        value_col=args.value_col,
        window=args.history_len,
        bins=args.volatility_bin_tiers,
        dayfirst=args.dayFirst,
    )
    volatility_bin_val = compute_volatility_bin(
        val_df,
        time_col=args.time_col,
        value_col=args.value_col,
        window=args.history_len,
        bins=args.volatility_bin_tiers,
        dayfirst=args.dayFirst,
    )
    volatility_bin_test = compute_volatility_bin(
        test_df,
        time_col=args.time_col,
        value_col=args.value_col,
        window=args.history_len,
        bins=args.volatility_bin_tiers,
        dayfirst=args.dayFirst,
    )

    # two-stage split (configurable via args.residual_base_frac / args.residual_base_epochs)
    base_frac = float(getattr(args, "residual_base_frac", 0.3))
    base_epochs, delta_epochs = split_two_stage_epochs(
        total_epochs=int(args.epochs),
        base_frac=base_frac,
        min_base=int(getattr(args, "residual_min_base_epochs", 1)),
        min_delta=int(getattr(args, "residual_min_delta_epochs", 1)),
    )
    if getattr(args, "residual_base_epochs", None) is not None:
        base_epochs = int(getattr(args, "residual_base_epochs"))
        base_epochs = max(0, min(base_epochs, int(args.epochs) - 1))
        delta_epochs = int(args.epochs) - base_epochs

    live_logger.info("-----------------------------------------------------")
    live_logger.info(f"Residual training enabled: BASE epochs={base_epochs}, DELTA epochs={delta_epochs}")
    live_logger.info("True-vs-shuffled is fully removed.")
    live_logger.info("-----------------------------------------------------")

    lora_cfg = {
        "r": int(args.lora_r),
        "alpha": int(args.lora_alpha),
        "dropout": float(args.lora_dropout),
        "target_modules": args.target_modules,
    }

    # ============ STAGE 1: Train BASE (force_no_news=True) ============
    live_logger.info("======== Stage 1/2: Training BASE (no news) ========")

    tokenizer, base_train_model = load_llama_lora(
        base_model=args.base_model,
        tokenizer_id=args.tokenizer,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        max_seq_len=args.max_seq_len,
        device=device,
        horizon=args.horizon,
        patch_dim=patch_len,
        patch_dropout=float(getattr(args, "patch_dropout", 0.0)),
        head_dropout=float(getattr(args, "head_dropout", 0.0)),
        head_mlp=bool(getattr(args, "head_mlp", False)),
    )
    base_train_model.to(device)

    optim_base = AdamW(
        filter(lambda p: p.requires_grad, base_train_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_batches = len(train_loader)
    total_opt_steps_base = math.ceil((num_batches * max(1, base_epochs) * args.rl_cycle_steps) / max(1, args.grad_accum))
    warmup_steps_base = int(getattr(args, "warmup_ratio", 0.1) * total_opt_steps_base)
    warmup_steps_base = min(warmup_steps_base, max(0, total_opt_steps_base - 1))

    scheduler_base = get_cosine_schedule_with_warmup(
        optim_base,
        num_warmup_steps=warmup_steps_base,
        num_training_steps=total_opt_steps_base,
    )

    # Base stage: RL is not meaningful (no-news). We keep tpl_id fixed.
    allowed_tpl_ids = sorted([t["id"] for t in templates.values()])
    tpl_id = allowed_tpl_ids[0]
    policy_name_base = "none"

    best_base_metric = float("inf")
    stale_rounds = 0
    loss_window = deque(maxlen=50)
    base_val_loss_per_epoch, base_mse_per_epoch, base_mae_per_epoch = [], [], []

    global_step = 0

    for epoch in range(base_epochs):
        pbar = tqdm(train_loader, desc=f"[BASE] Epoch {epoch+1}/{base_epochs}")

        for _, batch in enumerate(pbar):
            input_ids, attn, ts_patches, ts_patch_mask, targets_z, metas, _ = build_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                news_df=news_df,
                policy_name=policy_name_base,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                epoch=epoch,
                record_train_prompt=True,
                testing=False,
                force_no_news=True,
                news_dropout=False,
            )

            input_ids = input_ids.to(device)
            attn = attn.to(device)
            ts_patches = ts_patches.to(device)
            ts_patch_mask = ts_patch_mask.to(device)
            targets_z = targets_z.to(device)

            for _ in range(args.rl_cycle_steps):
                base_train_model.train()
                out = base_train_model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    ts_patches=ts_patches,
                    ts_patch_mask=ts_patch_mask,
                    targets=targets_z,
                )
                loss = out["loss"] / args.grad_accum
                loss.backward()

                loss_window.append(float(loss.detach().cpu()))
                if global_step % 10 == 0:
                    avg_train_loss = sum(loss_window) / len(loss_window)
                    pbar.set_postfix(train_loss=f"{avg_train_loss:.6f}")

                if (global_step + 1) % args.grad_accum == 0:
                    optim_base.step()
                    scheduler_base.step()
                    optim_base.zero_grad(set_to_none=True)

                global_step += 1

        val_loss, val_mse, val_mae = evaluate_metrics_single(
            model=base_train_model,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name=policy_name_base,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            testing=False,
            true_pred_csv_path=None,
            news_dropout=False,
            force_no_news=True,
        )
        base_val_loss_per_epoch.append(val_loss)
        base_mse_per_epoch.append(val_mse)
        base_mae_per_epoch.append(val_mae)

        if args.reward_metric == "loss":
            metric_now = val_loss
        elif args.reward_metric == "mse":
            metric_now = val_mse
        else:
            metric_now = val_mae

        live_logger.info(
            f"[BASE][EVAL] epoch={epoch+1} tpl_id={tpl_id} "
            f"val_loss(zMSE)={val_loss:.6f} val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
        )

        if metric_now < best_base_metric - 1e-6:
            best_base_metric = metric_now
            stale_rounds = 0
            best_base_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")
            if os.path.isfile(best_base_path):
                os.remove(best_base_path)

            save_checkpoint(
                best_base_path,
                tokenizer,
                base_train_model,
                base_model_id=args.base_model,
                tokenizer_id=args.tokenizer or args.base_model,
                lora_cfg=lora_cfg,
                optimizer=optim_base,
                scheduler=scheduler_base,
                epoch=epoch,
                global_step=global_step,
            )
            live_logger.info(f"[BASE] New best saved to {best_base_path} ({args.reward_metric}={best_base_metric:.6f})")
        else:
            stale_rounds += 1
            live_logger.info(f"[BASE] stale_rounds={stale_rounds}/{args.early_stop_patience} best={best_base_metric:.6f}")

        if stale_rounds >= args.early_stop_patience:
            live_logger.info(f"[BASE] Early stopping triggered at epoch {epoch+1}.")
            break

    # Free base training model memory (we will reload base best)
    del base_train_model
    gc.collect()
    torch.cuda.empty_cache()

    best_base_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")
    if not os.path.exists(best_base_path):
        raise FileNotFoundError(f"Base checkpoint not found: {best_base_path}")

    # ============ STAGE 2: Train DELTA (residual) ============
    live_logger.info("======== Stage 2/2: Training DELTA (news -> residual) ========")

    # teacher base (frozen)
    base_tok, base_model = load_checkpoint(
        best_base_path,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        device_map=None,
        is_trainable=False,
    )
    # keep tokenizer consistent
    tokenizer = base_tok
    base_model.to(device)
    freeze_module(base_model)

    # delta model init from base checkpoint but trainable
    _, delta_model = load_checkpoint(
        best_base_path,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        device_map=None,
        is_trainable=True,
    )
    delta_model.to(device)
    # important: start delta head near zero so it predicts small residual by default
    zero_regressor_head(delta_model)

    optim_delta = AdamW(
        filter(lambda p: p.requires_grad, delta_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_opt_steps_delta = math.ceil((len(train_loader) * max(1, delta_epochs) * args.rl_cycle_steps) / max(1, args.grad_accum))
    warmup_steps_delta = int(getattr(args, "warmup_ratio", 0.1) * total_opt_steps_delta)
    warmup_steps_delta = min(warmup_steps_delta, max(0, total_opt_steps_delta - 1))

    scheduler_delta = get_cosine_schedule_with_warmup(
        optim_delta,
        num_warmup_steps=warmup_steps_delta,
        num_training_steps=total_opt_steps_delta,
    )

    # RL setup for DELTA stage
    normalizer = RewardNormalizer(ema=args.reward_ema, use_group_norm=args.domain_reward_norm)
    val_state = ValidationState(ema_alpha=args.val_ema_alpha)
    context_vector = encode_instruction(args, ctx={}, volatility_bin=volatility_bin)

    tpl_features, feat_dim = make_tpl_feature_fn(
        templates=templates,
        add_one_hot=True,
        add_cost_proxy=False,
        add_cross_terms=True,
    )
    allowed_tpl_ids = sorted([t["id"] for t in templates.values()])

    d_tpl = len(context_vector) + len(tpl_features(allowed_tpl_ids[0], context_vector=context_vector))
    d_pol = len(context_vector)
    bandit_tpl = LinTS(d_tpl, v=args.ts_v) if args.rl_algo == "lints" else LinUCB(d_tpl, alpha=args.ucb_alpha)
    policy_space = ["keywords", "sentiment", "keyword_sentiment_hybrid"]
    bandit_pol = LinTS(d_pol, v=args.ts_v) if args.rl_algo == "lints" else LinUCB(d_pol, alpha=args.ucb_alpha)

    best_metric = float("inf")
    stale_rounds = 0
    loss_window = deque(maxlen=50)

    tpl_id = allowed_tpl_ids[0]
    policy_name = "none"
    best_tpl_id = tpl_id
    best_policy_name = policy_name

    val_loss_per_epoch, mse_loss_per_epoch, mae_loss_per_epoch = [], [], []

    def draw_metric_trend():
        p = f"./checkpoints/{args.taskName}"
        os.makedirs(p, exist_ok=True)
        epochs = list(range(1, len(val_loss_per_epoch) + 1))

        plt.figure()
        plt.plot(epochs, val_loss_per_epoch, label="Val Loss (z-MSE, combined)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss (combined z-space MSE)")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(p, f"ValLoss_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(epochs, mse_loss_per_epoch, label="Val MSE (raw, combined)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Validation MSE (combined raw scale)")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(p, f"ValMSE_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(epochs, mae_loss_per_epoch, label="Val MAE (raw, combined)")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Validation MAE (combined raw scale)")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(p, f"ValMAE_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()

        live_logger.info(f"Saved metric curves under {p}")

    def record_test_results_csv(mse, mae):
        try:
            p = f"./results"
            os.makedirs(p, exist_ok=True)
            csv_path = os.path.join(p, f"test_results.csv")
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Task", "MSE", "MAE"])
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([args.taskName, mse, mae])
            live_logger.info(f"Saved test results to {csv_path}")
        except Exception as e:
            live_logger.error(f"Failed to save test results to CSV: {e}")

    def draw_pred_true():
        try:
            p = f"./checkpoints/{args.taskName}"
            os.makedirs(p, exist_ok=True)
            df = pd.read_csv(true_pred_csv_path)
            plt.figure()
            plt.plot(df["true"], label="True Values")
            plt.plot(df["pred"], label="Predicted Values")
            plt.xlabel("Sample Index")
            plt.ylabel("Values")
            plt.title("Predicted and True Values (combined)")
            plt.legend()
            plt.grid(True)
            fig_path = os.path.join(p, f"PredVsTrue_{args.taskName}.png")
            plt.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close()
            live_logger.info(f"Saved Pred vs True plot to {fig_path}")
        except Exception as e:
            live_logger.error(f"Failed to draw Pred vs True plot: {e}")

    for epoch in range(delta_epochs):
        pbar = tqdm(train_loader, desc=f"[DELTA] Epoch {epoch+1}/{delta_epochs}")

        # epoch-level bandit selection (optional)
        if (args.select_policy_by == "epoch") and args.rl_use == 1:
            context_vector = get_context_features(
                None,
                news_df,
                args,
                prev_model_loss_n=None,
                prev_model_loss_ema_n=None,
                val_state=val_state,
                train_loader=train_loader,
                volatility_bin=volatility_bin,
            )

            tpl_id, policy_name, pol_idx = bandit_round_update_residual(
                base_model=base_model,
                delta_model=delta_model,
                tokenizer=tokenizer,
                probe_loader=val_loader,
                templates=templates,
                allowed_tpl_ids=allowed_tpl_ids,
                news_df=news_df,
                policy_space=policy_space,
                policy_kw=policy_kw,
                args=args,
                device=device,
                volatility_bin=volatility_bin_val,
                context_vector=context_vector,
                tpl_features=tpl_features,
                bandit_tpl=bandit_tpl,
                bandit_pol=bandit_pol,
                normalizer=normalizer,
                live_logger=live_logger,
                round_id=epoch + 1,
                bidx=None,
                global_step=global_step,
            )
            live_logger.info(f"[DELTA] EPOCH_BEGIN epoch={epoch+1}, tpl_id={tpl_id}, policy={policy_name}")

        for bidx, batch in enumerate(pbar):
            # batch-level bandit selection (optional)
            if (args.select_policy_by == "batch") and args.rl_use == 1:
                context_vector = get_context_features(
                    batch,
                    news_df,
                    args,
                    prev_model_loss_n=None,
                    prev_model_loss_ema_n=None,
                    val_state=val_state,
                    train_loader=train_loader,
                    volatility_bin=volatility_bin,
                )

                tpl_id, policy_name, pol_idx = bandit_round_update_residual(
                    base_model=base_model,
                    delta_model=delta_model,
                    tokenizer=tokenizer,
                    probe_loader=val_loader,
                    templates=templates,
                    allowed_tpl_ids=allowed_tpl_ids,
                    news_df=news_df,
                    policy_space=policy_space,
                    policy_kw=policy_kw,
                    args=args,
                    device=device,
                    volatility_bin=volatility_bin_val,
                    context_vector=context_vector,
                    tpl_features=tpl_features,
                    bandit_tpl=bandit_tpl,
                    bandit_pol=bandit_pol,
                    normalizer=normalizer,
                    live_logger=live_logger,
                    round_id=epoch * len(pbar) + bidx,
                    bidx=bidx,
                    global_step=global_step,
                )

            # build delta inputs (with news)
            ids_d, attn_d, ts_p, ts_pm, targets_z, metas, _ = build_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                news_df=news_df,
                policy_name=policy_name,
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                epoch=epoch,
                record_train_prompt=True,
                testing=False,
                force_no_news=False,
                news_dropout=False,
            )
            # build base text inputs (no news)
            ids_b, attn_b, _, _, _, _, _ = build_batch_inputs(
                batch=batch,
                tokenizer=tokenizer,
                templates=templates,
                tpl_id=tpl_id,
                args=args,
                news_df=news_df,
                policy_name="none",
                policy_kw=policy_kw,
                volatility_bin=volatility_bin,
                epoch=epoch,
                record_train_prompt=False,
                testing=False,
                force_no_news=True,
                news_dropout=False,
            )

            ids_d = ids_d.to(device)
            attn_d = attn_d.to(device)
            ids_b = ids_b.to(device)
            attn_b = attn_b.to(device)

            ts_p = ts_p.to(device)
            ts_pm = ts_pm.to(device)
            targets_z = targets_z.to(device)

            for _ in range(args.rl_cycle_steps):
                delta_model.train()

                # teacher base pred (no grad)
                with torch.no_grad():
                    out_base = base_model(
                        input_ids=ids_b,
                        attention_mask=attn_b,
                        ts_patches=ts_p,
                        ts_patch_mask=ts_pm,
                        targets=None,
                    )
                    base_pred = out_base["pred"]

                # residual target
                delta_targets = targets_z - base_pred.detach()

                out_delta = delta_model(
                    input_ids=ids_d,
                    attention_mask=attn_d,
                    ts_patches=ts_p,
                    ts_patch_mask=ts_pm,
                    targets=delta_targets,
                )
                loss = out_delta["loss"] / args.grad_accum
                loss.backward()

                loss_window.append(float(loss.detach().cpu()))
                if global_step % 10 == 0:
                    avg_train_loss = sum(loss_window) / len(loss_window)
                    pbar.set_postfix(train_loss=f"{avg_train_loss:.6f}")

                if (global_step + 1) % args.grad_accum == 0:
                    optim_delta.step()
                    scheduler_delta.step()
                    optim_delta.zero_grad(set_to_none=True)

                global_step += 1

        # end-of-epoch eval (combined)
        val_loss, val_mse, val_mae = evaluate_metrics_residual(
            base_model=base_model,
            delta_model=delta_model,
            tokenizer=tokenizer,
            data_loader=val_loader,
            templates=templates,
            tpl_id=tpl_id,
            args=args,
            news_df=news_df,
            policy_name=policy_name,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_val,
            testing=False,
            true_pred_csv_path=None,
            news_dropout=False,
        )
        val_loss_per_epoch.append(val_loss)
        mse_loss_per_epoch.append(val_mse)
        mae_loss_per_epoch.append(val_mae)

        live_logger.info(
            f"[DELTA][EVAL] epoch={epoch+1} tpl_id={tpl_id} policy={policy_name} "
            f"val_loss(zMSE)={val_loss:.6f} val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
        )

        # update val_state for context
        if args.reward_metric == "loss":
            val_state.update(val_loss)
            metric_now = val_loss
        elif args.reward_metric == "mse":
            val_state.update(val_mse)
            metric_now = val_mse
        else:
            val_state.update(val_mae)
            metric_now = val_mae

        if metric_now < best_metric - 1e-6:
            best_metric = metric_now
            stale_rounds = 0
            best_tpl_id = tpl_id
            best_policy_name = policy_name

            best_delta_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_delta_{args.taskName}")
            if os.path.isfile(best_delta_path):
                os.remove(best_delta_path)

            save_checkpoint(
                best_delta_path,
                tokenizer,
                delta_model,
                base_model_id=args.base_model,
                tokenizer_id=args.tokenizer or args.base_model,
                lora_cfg=lora_cfg,
                optimizer=optim_delta,
                scheduler=scheduler_delta,
                epoch=epoch,
                global_step=global_step,
            )

            # write a small meta pointer so testing knows how to load pair
            with open(os.path.join(f"./checkpoints/{args.taskName}", "residual_pair.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_base": f"best_base_{args.taskName}",
                        "best_delta": f"best_delta_{args.taskName}",
                        "best_tpl_id": int(best_tpl_id),
                        "best_policy_name": str(best_policy_name),
                        "reward_metric": str(args.reward_metric),
                        "best_metric": float(best_metric),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            live_logger.info(f"[DELTA] New best delta saved: {best_delta_path} ({args.reward_metric}={best_metric:.6f})")
        else:
            stale_rounds += 1
            live_logger.info(f"[DELTA] stale_rounds={stale_rounds}/{args.early_stop_patience} best={best_metric:.6f}")

        if stale_rounds >= args.early_stop_patience:
            live_logger.info(f"[DELTA] Early stopping triggered at epoch {epoch+1}.")
            break

        # log data statistics at first delta epoch
        if epoch == 0:
            live_logger.info("---------------------trainset and valset prompt statistics--------------------------------")
            live_logger.info(f"Prompt number: {dataStatistic.prompt_num}")
            live_logger.info(f"News number total: {dataStatistic.news_num_total}")
            live_logger.info(f"Max news number per prompt: {dataStatistic.max_news_num_per_prompt}")
            live_logger.info(f"Min news number per prompt: {dataStatistic.min_news_num_per_prompt}")
            live_logger.info(f"Mean news number per prompt: {dataStatistic.mean_news_num_per_prompt}")
            live_logger.info(f"Prompt with max news number: {dataStatistic.the_prompt_with_most_news_num}")
            live_logger.info(f"Prompt with max news length: {dataStatistic.prompt_with_max_news_len}")
            live_logger.info(f"Prompt with max total length: {dataStatistic.prompt_with_max_total_len}")
            live_logger.info("-----------------------------------------------------")

    # end of training
    draw_metric_trend()
    dataStatistic.clear()

    # ============ TEST (combined with best pair) ============
    if test_loader is not None:
        gc.collect()
        torch.cuda.empty_cache()

        best_base_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_base_{args.taskName}")
        best_delta_path = os.path.join(f"./checkpoints/{args.taskName}", f"best_delta_{args.taskName}")

        tok_b, base_best = load_checkpoint(best_base_path, args.load_in_4bit, args.gradient_checkpointing, None, False)
        tok_d, delta_best = load_checkpoint(best_delta_path, args.load_in_4bit, args.gradient_checkpointing, None, False)

        # enforce same tokenizer reference
        tokenizer = tok_d
        base_best.to(device)
        delta_best.to(device)
        freeze_module(base_best)

        live_logger.info("Loaded best BASE + DELTA for testing (combined).")

        # choose best tpl/policy if RL used, else current
        tpl_for_test = best_tpl_id if args.rl_use == 1 else tpl_id
        pol_for_test = best_policy_name if args.rl_use == 1 else policy_name

        test_loss, test_mse, test_mae = evaluate_metrics_residual(
            base_model=base_best,
            delta_model=delta_best,
            tokenizer=tokenizer,
            data_loader=test_loader,
            templates=templates,
            tpl_id=tpl_for_test,
            args=args,
            news_df=news_df,
            policy_name=pol_for_test,
            policy_kw=policy_kw,
            device=device,
            volatility_bin=volatility_bin_test,
            testing=True,
            true_pred_csv_path=true_pred_csv_path,
            news_dropout=False,
        )

        live_logger.info("---------------------testset prompt statistics--------------------------------")
        live_logger.info(f"Prompt number: {dataStatistic.prompt_num}")
        live_logger.info(f"News number total: {dataStatistic.news_num_total}")
        live_logger.info(f"Max news number per prompt: {dataStatistic.max_news_num_per_prompt}")
        live_logger.info(f"Min news number per prompt: {dataStatistic.min_news_num_per_prompt}")
        live_logger.info(f"Mean news number per prompt: {dataStatistic.mean_news_num_per_prompt}")
        live_logger.info(f"Prompt with max news number: {dataStatistic.the_prompt_with_most_news_num}")
        live_logger.info(f"Prompt with max news length = {dataStatistic.newslen_prompt_with_max_news_len}: {dataStatistic.prompt_with_max_news_len}")
        live_logger.info(f"Prompt with max total length = {len(dataStatistic.prompt_with_max_total_len)}: {dataStatistic.prompt_with_max_total_len}")
        live_logger.info("-----------------------------------------------------")

        tqdm.write(f"[TEST][COMBINED] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
        live_logger.info(f"[TEST][COMBINED] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")

        record_test_results_csv(test_mse, test_mae)
        draw_pred_true()
