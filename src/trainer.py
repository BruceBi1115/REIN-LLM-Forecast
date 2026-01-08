# trainer.py (REGRESSION version - full file)

import csv
import os, json
import math
from collections import deque

import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup

from .utils.utils import set_seed, device_from_id
from .data_construction.data import make_loader
from .news_rules import load_news, get_candidates, select_news, _load_keywords
from .data_construction.prompt import load_templates, format_news, build_prompt
from .RL.rl_bandit import LinTS, LinUCB, RewardNormalizer
from .ValidationState import ValidationState
from .utils.logger import setup_live_logger
from .RL.features import bandit_select, get_context_features, encode_instruction

from .model import load_llama_lora


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
    p = args.news_dropout
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
        # pad one patch
        p = np.zeros((1, patch_len), dtype=np.float32)
        p[0, :L] = x
        m = np.ones((1,), dtype=np.int64)
        return p, m

    idxs = list(range(0, L - patch_len + 1, stride))
    patches = np.stack([x[i : i + patch_len] for i in idxs], axis=0).astype(np.float32)  # (P, patch_len)
    mask = np.ones((patches.shape[0],), dtype=np.int64)
    return patches, mask


def history_text(history_z: list[float], mu: float, sigma: float) -> str:
    """
    Keep a brief textual history (optional). Main signal is via patch embeddings.
    """

    history_z_str = ", ".join([f"{v:.4f}" for v in history_z])


    return (
        f"History is z-scored using window mean={mu:.4f}, std={sigma:.4f}. "
        f"Last {len(history_z)} z-values: {history_z_str}"
    )


def forward_batch_build_inputs(
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
    testing = False,
):
    """
    Build:
      - prompt input_ids / attn (text only)
      - ts_patches (z-scored history patches)
      - patch_mask
      - targets_z (z-scored targets)
      - metas (mu/sigma + prompt text for debugging)
    """
    L, H = int(args.history_len), int(args.horizon)
    hist_budget = int(args.token_budget * args.token_budget_history_frac)
    news_budget = int(args.token_budget * args.token_budget_news_frac)

    patch_len = int(getattr(args, "patch_len", 4))
    patch_stride = int(getattr(args, "patch_stride", patch_len))


    tpl_text = templates[tpl_id]["text"]

    input_ids_list = []
    attn_list = []

    patches_list = []
    patchmask_list = []

    targets_list = []
    metas = []
    prompt_texts = []

    for i in range(len(batch["history_value"])):
        history = batch["history_value"][i].tolist()
        target = batch["target_value"][i].tolist()
        t_target = batch["target_time"][i]

        cand = get_candidates(news_df, args.news_time_col, t_target, args.news_window_days, args.news_topM)
        selected = select_news(cand, policy_name, args.news_text_col, policy_kw, args.news_topK)

        # z-score based on this sample's history window
        mu, sigma = _zstats(history, eps=float(getattr(args, "zscore_eps", 1e-6)))
        history_z = _zscore(history, mu, sigma)
        target_z = _zscore(target, mu, sigma)

        # patches from z-scored history (main numeric signal)
        p, pm = _make_patches(history_z, patch_len=patch_len, stride=patch_stride)

        # news string (text)
        news_str = format_news(
            selected,
            args.news_text_col,
            news_budget,
            tokenizer,
            summary_method=args.news_summary_method,
            max_sentences=args.news_max_sentences,
        )
        if not testing:
            news_str = _maybe_news_dropout(news_str, args)

        start_date = batch["history_times"][0][i]
        end_date = batch["history_times"][-1][i]
        prediction_start = batch["target_times"][0][i]
        prediction_end = batch["target_times"][-1][i]

        # brief history text only (do NOT dump full numeric list)
        hist_str = history_text(history_z, mu, sigma)

        # build prompt text (NO output slots, NO target appended)
        prompt = build_prompt(
            tpl_text,
            L,
            H,
            args.unit,
            args.description,
            hist_str,
            news_str,
            start_date=start_date,
            end_date=end_date,
            freq=args.freq_min,
            value_col=args.value_col,
            pred_end=prediction_end,
            pred_start=prediction_start,
            region=args.region,
        )

        # optional: keep a small output-format instruction, but model will regress from hidden states anyway
        prompt = (
            prompt
            + "\n\n[Output]\n"
            + f"Predict the next {H} steps (internally as z-values). Do not output numbers.\n"
        )

        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=int(args.max_seq_len),
            return_attention_mask=True,
        )
        ids = enc["input_ids"]
        am = enc["attention_mask"]

        input_ids_list.append(ids)
        attn_list.append(am)

        patches_list.append(p)       # (P_i, patch_len)
        patchmask_list.append(pm)    # (P_i,)

        targets_list.append(np.asarray(target_z, dtype=np.float32))  # (H,)

        meta = {"mu": mu, "sigma": sigma}
        metas.append(meta)
        prompt_texts.append(prompt)

        if record_train_prompt:
            ckpt_dir = os.path.join("./checkpoints", args.taskName)
            os.makedirs(ckpt_dir, exist_ok=True)
            prompt_path = os.path.join(ckpt_dir, f"prompts_{args.taskName}.jsonl")
            with open(prompt_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "batch_idx": i,
                            "epoch_num": epoch + 1,
                            "template_id": tpl_id,
                            "prompt": prompt,
                            "mu": mu,
                            "sigma": sigma,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # pad text
    max_t = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id

    B = len(input_ids_list)
    input_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_t), dtype=torch.long)

    for i, (ids, am) in enumerate(zip(input_ids_list, attn_list)):
        t = len(ids)
        input_ids[i, :t] = torch.tensor(ids, dtype=torch.long)
        attn[i, :t] = torch.tensor(am, dtype=torch.long)

    # pad patches
    max_p = max(p.shape[0] for p in patches_list)
    patch_len = int(getattr(args, "patch_len", 4))

    ts_patches = torch.zeros((B, max_p, patch_len), dtype=torch.float32)
    ts_patch_mask = torch.zeros((B, max_p), dtype=torch.long)

    for i, (p, pm) in enumerate(zip(patches_list, patchmask_list)):
        P_i = p.shape[0]
        ts_patches[i, :P_i, :] = torch.tensor(p, dtype=torch.float32)
        ts_patch_mask[i, :P_i] = torch.tensor(pm, dtype=torch.long)

    targets_z = torch.stack([torch.tensor(t, dtype=torch.float32) for t in targets_list], dim=0)  # (B, H)

    return input_ids, attn, ts_patches, ts_patch_mask, targets_z, metas, prompt_texts


@torch.no_grad()
def evaluate_metrics(
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
):
    """
    Returns:
      - loss_avg: z-space MSE loss (training objective)
      - mse_avg: original-scale MSE
      - mae_avg: original-scale MAE
    """
    model.eval()

    loss_sum, n_samples = 0.0, 0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0

    if testing:
        ckpt_dir = os.path.join("./checkpoints", args.taskName)
        os.makedirs(ckpt_dir, exist_ok=True)
        ans_jsonl_path = os.path.join(ckpt_dir, f"test_answers_{args.taskName}.jsonl")

    for bidx, batch in enumerate(data_loader):
        input_ids, attn, ts_patches, ts_patch_mask, targets_z, metas, prompt_texts = forward_batch_build_inputs(
            batch,
            tokenizer,
            templates,
            tpl_id,
            args,
            news_df,
            policy_name,
            policy_kw,
            volatility_bin=volatility_bin,
            epoch=-1,
            record_train_prompt=False,
            testing=testing,
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
        pred_z = out["pred"]  # (B, H)

        bs = input_ids.size(0)
        loss_sum += float(loss.detach().cpu()) * bs
        n_samples += bs

        pred_z_cpu = pred_z.detach().to(torch.float32).cpu().numpy()
        targets_cpu = batch["target_value"].detach().cpu().numpy()  # (B, H) raw scale

        for i in range(bs):
            mu = float(metas[i]["mu"])
            sigma = float(metas[i]["sigma"])

            pred_denorm = _inv_zscore(pred_z_cpu[i].tolist(), mu, sigma)  # list H
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
                with open(ans_jsonl_path, "a", encoding="utf-8") as f:
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


def bandit_round_update(
    model,
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
    model.eval()
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

    probe_loss, probe_mse, probe_mae = evaluate_metrics(
        model,
        tokenizer,
        probe_loader,
        templates,
        tpl_id,
        args,
        news_df,
        policy_name,
        policy_kw,
        device,
        volatility_bin=volatility_bin,
        testing=False,
        true_pred_csv_path=None,
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
        f"BANDIT_ROUND round={round_id} tpl_id={tpl_id} policy={policy_name} "
        f"probe_loss={probe_loss:.6f} probe_mse={probe_mse:.6f} probe_mae={probe_mae:.6f} "
        f"reward_norm={r_hat:.6f}"
    )

    return tpl_id, policy_name, pol_idx


def main(args):
    filename = "log_rl_" + str(args.rl_use) + "_epoch_" + str(args.epochs) + "_" + args.taskName
    log_filename = filename + ".log"
    live_logger, live_path, log_jsonl = setup_live_logger(
        save_dir=args.save_dir + "/" + args.taskName, filename=log_filename
    )
    print(f"[live log] {live_path}  (实时查看: tail -f '{live_path}')")

    # clean outputs
    ckpt_dir = os.path.join("./checkpoints", args.taskName)
    os.makedirs(ckpt_dir, exist_ok=True)

    prompt_path = os.path.join(ckpt_dir, f"prompts_{args.taskName}.jsonl")
    with open(prompt_path, "w", encoding="utf-8"):
        pass

    ans_jsonl_path = os.path.join(ckpt_dir, f"test_answers_{args.taskName}.jsonl")
    with open(ans_jsonl_path, "w", encoding="utf-8"):
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

    news_df = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])
    news_df[args.news_time_col] = pd.to_datetime(news_df[args.news_time_col], dayfirst=args.dayFirst)
    if args.news_path:
        news_df = load_news(args.news_path, args.news_time_col, args.news_tz)

    policy_kw = _load_keywords(args.keyword_path)
    templates = load_templates(args.template_pool)

    patch_len = int(getattr(args, "patch_len", 4))

    tokenizer, model = load_llama_lora(
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
    model.to(device)

    optim = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_batches = len(train_loader)
    total_opt_steps = math.ceil((num_batches * args.epochs * args.rl_cycle_steps) / max(1, args.grad_accum))
    warmup_steps = int(getattr(args, "warmup_ratio", 0.1) * total_opt_steps)

    assert args.lr > 0, f"args.lr is {args.lr}"
    assert total_opt_steps > 0, f"total_opt_steps is {total_opt_steps}"
    assert warmup_steps < total_opt_steps, f"warmup_steps >= total_opt_steps ({warmup_steps} >= {total_opt_steps})"

    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

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

    live_logger.info("-----------------------------------------------------")
    live_logger.info(
        f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}"
    )
    live_logger.info(
        f"Training started with volatility bins: trainset={volatility_bin}, valset={volatility_bin_val}, testset={volatility_bin_test}"
    )
    live_logger.info(f"Volatility bin tiers: {args.volatility_bin_tiers}")
    live_logger.info("-----------------------------------------------------")
    live_logger.info(
        f"RL settings: rl_use={args.rl_use}, rl_algo={args.rl_algo}, reward_metric={args.reward_metric}, reward_mode={args.reward_mode}, "
        f"select_policy_by={args.select_policy_by}, rl_cycle_steps={args.rl_cycle_steps}, rl_update_times={args.rl_update_times}"
    )
    live_logger.info("-----------------------------------------------------")
    live_logger.info(f"Templates loaded: {len(templates)} templates from {args.template_pool}")
    live_logger.info(f"news_topK={args.news_topK}, news_topM={args.news_topM}, news_window_days={args.news_window_days}")
    live_logger.info(
        f"News retrieval policy keywords loaded from: {args.keyword_path}, total number of policy keywords: {len(policy_kw)}"
    )
    live_logger.info(f"Epochs: {args.epochs}, Max Steps: {args.max_steps}, Early Stop Patience: {args.early_stop_patience}")
    live_logger.info(
        f"Base model: {args.base_model}, LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, target_modules={args.target_modules}"
    )
    live_logger.info(
        f"Max seq len: {args.max_seq_len}, History len: {args.history_len}, Horizon: {args.horizon}, Stride: {args.stride}"
    )
    live_logger.info(f"Patch len: {getattr(args, 'patch_len', 4)}, Patch stride: {getattr(args, 'patch_stride', getattr(args,'patch_len',4))}")
    live_logger.info(f"Description: {args.description}")
    live_logger.info(f"news_dropout={getattr(args, 'news_dropout', 0.0)}")
    live_logger.info("-----------------------------------------------------")
    live_logger.info(f"Device: {device}, Model dtype: {next(model.parameters()).dtype}")
    live_logger.info(f"Optimizer: AdamW, LR: {args.lr}, Weight Decay: {args.weight_decay}")
    live_logger.info(f"Scheduler: Cosine with Warmup, Total Steps: {total_opt_steps}, Warmup Steps: {warmup_steps}")
    live_logger.info(
        f"Batch size: {args.batch_size}, Gradient Accumulation: {args.grad_accum}, Effective Batch Size: {args.batch_size * args.grad_accum}"
    )
    live_logger.info(
        f"Token budget: {args.token_budget} (history frac: {args.token_budget_history_frac}, news frac: {args.token_budget_news_frac})"
    )
    live_logger.info("-----------------------------------------------------")

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

    global_step = 0
    best_metric = float("inf")
    stale_rounds = 0
    loss_window = deque(maxlen=50)

    tpl_id = allowed_tpl_ids[0]
    policy_name = policy_space[0]

    val_loss_per_epoch = []
    mse_loss_per_epoch = []
    mae_loss_per_epoch = []

    def draw_metric_trend():
        p = f"./checkpoints/{args.taskName}"
        os.makedirs(p, exist_ok=True)
        epochs = list(range(1, len(val_loss_per_epoch) + 1))
        xlabel = "Epoch"

        plt.figure()
        plt.plot(epochs, val_loss_per_epoch, label="Val Loss (z-MSE)")
        plt.xlabel(xlabel)
        plt.ylabel("Loss")
        plt.title("Validation Loss (z-space MSE)")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(p, f"ValLoss_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        live_logger.info(f"Saved loss curve to {fig_path}")

        plt.figure()
        plt.plot(epochs, mse_loss_per_epoch, label="Val MSE (raw)")
        plt.xlabel(xlabel)
        plt.ylabel("MSE")
        plt.title("Validation MSE (raw scale)")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(p, f"ValMSE_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        live_logger.info(f"Saved validation MSE curve to {fig_path}")

        plt.figure()
        plt.plot(epochs, mae_loss_per_epoch, label="Val MAE (raw)")
        plt.xlabel(xlabel)
        plt.ylabel("MAE")
        plt.title("Validation MAE (raw scale)")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(p, f"ValMAE_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        live_logger.info(f"Saved validation MAE curve to {fig_path}")

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
            plt.title("Predicted and True Values")
            plt.legend()
            plt.grid(True)
            fig_path = os.path.join(p, f"PredVsTrue_{args.taskName}.png")
            plt.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close()
            live_logger.info(f"Saved Pred vs True plot to {fig_path}")
        except Exception as e:
            live_logger.error(f"Failed to draw Pred vs True plot: {e}")

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

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

            tpl_id, policy_name, pol_idx = bandit_round_update(
                model=model,
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

            live_logger.info(f"EPOCH_BEGIN epoch={epoch+1}, selected_template_id={tpl_id}, selected_policy={policy_name}")

        # training batches
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

                tpl_id, policy_name, pol_idx = bandit_round_update(
                    model=model,
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

            input_ids, attn, ts_patches, ts_patch_mask, targets_z, metas, _ = forward_batch_build_inputs(
                batch,
                tokenizer,
                templates,
                tpl_id,
                args,
                news_df,
                policy_name,
                policy_kw,
                volatility_bin=volatility_bin,
                epoch=epoch,
                record_train_prompt=True,
            )

            input_ids = input_ids.to(device)
            attn = attn.to(device)
            ts_patches = ts_patches.to(device)
            ts_patch_mask = ts_patch_mask.to(device)
            targets_z = targets_z.to(device)

            for _ in range(args.rl_cycle_steps):
                model.train()
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    ts_patches=ts_patches,
                    ts_patch_mask=ts_patch_mask,
                    targets=targets_z,
                )
                loss = out["loss"]
                loss = loss / args.grad_accum
                loss.backward()

                loss_window.append(float(loss.detach().cpu()))
                if global_step % 10 == 0:
                    avg_train_loss = sum(loss_window) / len(loss_window)
                    pbar.set_postfix(train_loss=f"{avg_train_loss:.6f}")

                if (global_step + 1) % args.grad_accum == 0:
                    optim.step()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)

                global_step += 1

        # end-of-epoch eval (now cheap and stable; no generate/parsing)
        val_loss, val_mse, val_mae = evaluate_metrics(
            model,
            tokenizer,
            val_loader,
            templates,
            tpl_id,
            args,
            news_df,
            policy_name,
            policy_kw,
            device,
            volatility_bin=volatility_bin_val,
            testing=False,
            true_pred_csv_path=None,
        )
        val_loss_per_epoch.append(val_loss)
        mse_loss_per_epoch.append(val_mse)
        mae_loss_per_epoch.append(val_mae)

        live_logger.info(
            f"EVAL epoch={epoch+1} tpl_id={tpl_id} policy={policy_name} "
            f"val_loss(zMSE)={val_loss:.6f} val_mse(raw)={val_mse:.6f} val_mae(raw)={val_mae:.6f}"
        )

        # update state for bandit context if you use it
        if args.reward_metric == "loss":
            val_state.update(val_loss)
        elif args.reward_metric == "mse":
            val_state.update(val_mse)
        else:
            val_state.update(val_mae)

        # early stopping on reward metric
        if args.reward_metric == "loss":
            metric_now = val_loss
        elif args.reward_metric == "mse":
            metric_now = val_mse
        else:
            metric_now = val_mae

        if metric_now < best_metric - 1e-6:
            best_metric = metric_now
            stale_rounds = 0
        else:
            stale_rounds += 1
            live_logger.info(f"[Early Stop] stale_rounds={stale_rounds}/{args.early_stop_patience} best={best_metric:.6f}")

        if stale_rounds >= args.early_stop_patience:
            live_logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break

    draw_metric_trend()

    if test_loader is not None:
        test_loss, test_mse, test_mae = evaluate_metrics(
            model,
            tokenizer,
            test_loader,
            templates,
            tpl_id,
            args,
            news_df,
            policy_name,
            policy_kw,
            device,
            volatility_bin=volatility_bin_test,
            testing=True,
            true_pred_csv_path=true_pred_csv_path,
        )
        live_logger.info("-----------------------------------------------------")
        tqdm.write(f"[TEST] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
        live_logger.info(f"[TEST] loss(zMSE)={test_loss:.6f} mse(raw)={test_mse:.6f} mae(raw)={test_mae:.6f}")
        record_test_results_csv(test_mse, test_mae)
        draw_pred_true()
