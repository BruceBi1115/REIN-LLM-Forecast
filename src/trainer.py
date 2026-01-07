# trainer.py (GENERATIVE version - full file)

import csv
import os, json
import re
import math
import logging
from logging.handlers import WatchedFileHandler
from collections import deque
from itertools import islice

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
from .data_construction.prompt import load_templates, format_history, format_news, build_prompt
from .RL.rl_bandit import LinTS, LinUCB, RewardNormalizer
from .utils.metrics import rmse, mae, smape
from .model import build_pred_slots, load_llama_lora
from .ValidationState import ValidationState
from .utils.logger import setup_live_logger
from .RL.features import bandit_select, get_context_features, encode_instruction
from .model import END_TOKEN  # 如果你愿意就 import；不想 import 就直接写 "<END>"

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

def _short(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def _maybe_news_dropout(news_str: str, args) -> str:
    """
    随机丢弃 news，迫使模型在部分样本上不能依赖新闻走捷径。
    需要 args.news_dropout (0~1)，未提供则默认 0.
    """
    p = float(getattr(args, "news_dropout", 0.0) or 0.0)
    if p <= 0:
        return news_str
    if np.random.rand() < p:
        return ""
    return news_str


def _format_target_numbers(target_list, precision: int = 3) -> str:
    # 固定小数位，减少格式漂移
    return ",".join([f"{float(x):.{precision}f}" for x in target_list])


def _parse_first_h_numbers(text: str, H: int):
    if not text:
        return None

    # 只解析 FINAL: 后面的内容；找不到就退化为解析全体（便于排查）
    m = re.search(r"\bFINAL\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    region = m.group(1) if m else text

    # 支持整数/小数/科学计数法，如 1e-3
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", region)

    vals = []
    for s in nums:
        try:
            vals.append(float(s))
        except:
            continue
        if len(vals) >= H:
            break
    return vals if len(vals) >= H else None


@torch.no_grad()
def _generate_pred_numbers(model, tokenizer, prompt_input_ids, prompt_attn, H: int, device,
                           max_new_tokens: int = 256):
    end_id = tokenizer.convert_tokens_to_ids("<END>")
    if end_id is None or end_id == tokenizer.unk_token_id:
        end_id = tokenizer.eos_token_id  # 兜底，但正常不会走到这里

    gen = model.generate(
        input_ids=prompt_input_ids.to(device),
        attention_mask=prompt_attn.to(device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=end_id,   # <- 关键：遇到 <END> 就停
    )

    new_tokens = gen[:, prompt_input_ids.size(1):]
    txt = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
    # print("generated txt END_TOKEN ID: ", tokenizer.convert_tokens_to_ids("<END>"))

    # txt 里可能不含 <END>（被 skip_special_tokens 去掉了），但生成已被它停止
    vals = _parse_first_h_numbers(txt, H)
    return vals, txt


# 把一个 batch 的结构化样本 → prompt 文本 + target 文本 → tokenizer → (input_ids, labels_token_ids)
def forward_batch_build_inputs(batch, tokenizer, templates, tpl_id, args,
                               news_df, policy_name, policy_kw, news_encoder, volatility_bin,
                               epoch=-1, record_prompt=False):
    L, H = args.history_len, args.horizon
    hist_budget = int(args.token_budget * args.token_budget_history_frac)
    news_budget = int(args.token_budget * args.token_budget_news_frac)

    tpl_text = templates[tpl_id]['text']

    input_ids_list = []
    labels_list = []
    prompt_lens = []
    metas = []

    prec = args.target_precision

    for i in range(len(batch['history_value'])):
        history = batch['history_value'][i].tolist()
        target = batch['target_value'][i].tolist()
        t_target = batch['target_time'][i]
        series_id = batch['series_id'] if isinstance(batch['series_id'], list) else batch['series_id'] 
        cand = get_candidates(news_df, args.news_time_col, t_target, args.news_window_days, args.news_topM) 
        selected = select_news( cand, policy_name, args.news_text_col, policy_kw, args.news_topK )

        # === z-score based on THIS sample's history window ===
        mu, sigma = _zstats(history, eps=float(getattr(args, "zscore_eps", 1e-6)))
        history_z = _zscore(history, mu, sigma)
        target_z  = _zscore(target,  mu, sigma)

        # history/news 字符串
        hist_str = format_history(history_z, "z", hist_budget, tokenizer)  # unit 改成 z 更直观
        news_str = format_news(selected, args.news_text_col, news_budget, tokenizer, summary_method=args.news_summary_method, max_sentences=args.news_max_sentences )

        news_str = _maybe_news_dropout(news_str, args) 
        start_date = batch['history_times'][0][i] 
        end_date = batch['history_times'][-1][i] 
        prediction_start = batch['target_times'][0][i] 
        prediction_end = batch['target_times'][-1][i]

        # prompt 里最好明确说明“现在输入输出都是 z 值”
        prompt = build_prompt( tpl_text, L, H, args.unit, args.description, hist_str, news_str, start_date=start_date, end_date=end_date, freq=args.freq_min, value_col=args.value_col, pred_end=prediction_end, pred_start=prediction_start, region=args.region )
        # 输出格式（你可以继续用你原来的，也可以换 FINAL 单行；这里不强制）
        prompt = (
            prompt
            + "\n\n[Output Format]\n"
            + f"Generate exactly {H} numbers (z-values) separated by spaces.\n"
            + build_pred_slots(H) + "\n"
        )

        # target 用 z 值训练
        target_text = _format_target_numbers(target_z, precision=prec)
        full_text = prompt + target_text + "<END>\n"

        # target_text = "FINAL: " + _format_target_numbers(target, precision=prec) + f"<END>\n"
        # assert len(target) == H, "Target text length mismatch"
        
        # full_text = prompt + target_text + "\n"

        # token ids
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_seq_len
        ).input_ids

        pl = min(len(prompt_ids), len(full_ids))
        labels = [-100] * pl + full_ids[pl:]
        labels = labels[:len(full_ids)]

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        prompt_lens.append(pl)

        # print("Prompt preparation END_TOKEN ID: ", tokenizer.convert_tokens_to_ids("<END>"))

        if record_prompt:
            ckpt_dir = os.path.join("./checkpoints", args.taskName)
            os.makedirs(ckpt_dir, exist_ok=True)
            prompt_path = os.path.join(ckpt_dir, f"prompts_{args.taskName}.json")
            with open(prompt_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "batch_idx": i,
                        "epoch_num": epoch + 1,
                        "template_id": tpl_id,
                        # "series_id": series_id,
                        "prompt": full_text,
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write(",\n")

        metas.append({
            # "series_id": series_id,
            "mu": mu,
            "sigma": sigma,
        })

    # pad batch
    max_len = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((len(input_ids_list), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(labels_list), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)

    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        L_i = len(ids)
        input_ids[i, :L_i] = torch.tensor(ids, dtype=torch.long)
        labels[i, :L_i] = torch.tensor(lab, dtype=torch.long)
        attn[i, :L_i] = 1

    prompt_lens = torch.tensor(prompt_lens, dtype=torch.long)

    return input_ids, attn, labels, prompt_lens, metas


def evaluate_metrics(model, tokenizer, data_loader, templates, tpl_id, args,
                     news_df, policy_name, policy_kw, device, volatility_bin, testing=False, true_pred_csv_path=None):
    # 记录解析失败的样本数
    parse_fail = 0
    parse_total = 0

    model.eval()
    loss_sum, n_samples = 0.0, 0
    se_sum, ae_sum, n_elems = 0.0, 0.0, 0

    # 只对 data_loader 的前多少个 batch 做 generate() 评估。
    # max_gen_batches = int(getattr(args, "eval_gen_batches", 5))


    # 在每个被评估的 batch 里，最多对多少个样本做 generate
    gen_per_batch = int(getattr(args, "eval_gen_per_batch", 1))
    # model.generate() 最多允许生成多少个新 token
    gen_max_new_tokens = int(getattr(args, "gen_max_new_tokens", 512))

    # need_numeric = (args.reward_metric in ["mse", "mae"])

    with torch.no_grad():
        for bidx, batch in enumerate(data_loader):
            input_ids, attn, labels, prompt_lens, metas = forward_batch_build_inputs(
                batch, tokenizer, templates, tpl_id, args,
                news_df, policy_name, policy_kw,
                news_encoder=None, volatility_bin=volatility_bin
            )
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            batch_loss = out.loss
            bs = input_ids.size(0)

            loss_sum += float(batch_loss.detach().cpu()) * bs
            n_samples += bs


            take = min(bs, gen_per_batch)
            for i in range(take):
                pl = int(prompt_lens[i].item())
                prompt_ids = input_ids[i:i+1, :pl]
                prompt_attn = attn[i:i+1, :pl]

                pred_vals, txt = _generate_pred_numbers(
                    model, tokenizer, prompt_ids, prompt_attn,
                    H=args.horizon, device=device, max_new_tokens=gen_max_new_tokens
                )

                parse_total += 1
                if pred_vals is None:
                    parse_fail += 1

                if testing:
                    # === 记录到 json 文件（jsonl：每条一行，最稳）===
                    ckpt_dir = os.path.join("./checkpoints", args.taskName)
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ans_jsonl_path = os.path.join(ckpt_dir, f"test_answers_{args.taskName}.jsonl")
                    prompt_txt = tokenizer.decode(
                        prompt_ids[0].detach().cpu().tolist(),
                        skip_special_tokens=False
                    )
                    
                    record = {
                        "test_prompt": prompt_txt,  
                        "test_answer": txt,
                        "parsed so far": parse_total,
                        "failed so far": parse_fail,
                    }

                    with open(ans_jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # print("Test sample generated answer saved.")

                    

                if pred_vals is None:
                    se_sum += 1e6 * args.horizon
                    ae_sum += 1e6 * args.horizon
                    n_elems += args.horizon
                    continue
                # if pred_vals is None:
                #     # 不计入 mse/mae，只统计失败率
                #     continue


                mu = float(metas[i].get("mu", 0.0))
                sigma = float(metas[i].get("sigma", 1.0))

                # pred_vals 是 z 值，先反标准化
                pred_denorm = _inv_zscore(pred_vals[:args.horizon], mu, sigma)

                true_vals = batch["target_value"][i].detach().cpu().numpy().reshape(-1).tolist()
                true_vals = [float(x) for x in true_vals[:args.horizon]]

                pred = torch.tensor(pred_denorm, dtype=torch.float32)
                true = torch.tensor(true_vals, dtype=torch.float32)
                if true_pred_csv_path is not None:
                    with open(true_pred_csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(zip(pred_denorm, true_vals))

                se_sum += float(((pred - true) ** 2).sum().item())
                ae_sum += float((pred - true).abs().sum().item())
                n_elems += args.horizon

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
        tpl_by_id = {int(t['id']): t for t in templates}

    tpl_ids = sorted(tpl_by_id.keys())
    T = len(tpl_ids)

    id2idx = {tid: i for i, tid in enumerate(tpl_ids)}
    I = np.eye(T, dtype=np.float32)

    tpl_list = [tpl_by_id[tid] for tid in tpl_ids]
    n_paths_list = [float(t.get('n_paths', 1) or 1) for t in tpl_list]
    max_n_paths = max(n_paths_list) if n_paths_list else 1.0

    raw_breath_intensity = []
    for t in tpl_list:
        hb = float(bool(t.get('has_breath', False)))
        bf = float(t.get('breath_freq', 0) or 0)
        raw_breath_intensity.append(hb * (1.0 / bf) if hb > 0 and bf > 0 else 0.0)

    bi_min = min(raw_breath_intensity) if raw_breath_intensity else 0.0
    bi_max = max(raw_breath_intensity) if raw_breath_intensity else 1.0
    bi_range = (bi_max - bi_min) if bi_max > bi_min else 1.0

    def _cost_proxy(t):
        he = float(bool(t.get('has_example', False)))
        hb = float(bool(t.get('has_breath', False)))
        hd = float(bool(t.get('has_decomp', False)))
        hsc = float(bool(t.get('has_self_consistency', False)))
        np_norm = float(t.get('n_paths', 1) or 1) / max_n_paths
        return 0.4 * he + 0.5 * hd + 1.0 * hsc + 0.6 * np_norm + 0.2 * hb

    raw_costs = [_cost_proxy(t) for t in tpl_list]
    c_min = min(raw_costs) if raw_costs else 0.0
    c_max = max(raw_costs) if raw_costs else 1.0
    c_range = (c_max - c_min) if c_max > c_min else 1.0

    def _single_tpl_vec(tid: int) -> np.ndarray:
        t = tpl_by_id[int(tid)]

        he = float(bool(t.get('has_example', False)))
        hb = float(bool(t.get('has_breath', False)))
        hd = float(bool(t.get('has_decomp', False)))
        hsc = float(bool(t.get('has_self_consistency', False)))
        np_norm = float(t.get('n_paths', 1) or 1) / max_n_paths

        bf = float(t.get('breath_freq', 0) or 0)
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


def bandit_round_update(model, tokenizer, probe_loader,
                        templates, allowed_tpl_ids,
                        news_df, policy_space, policy_kw,
                        args, device, volatility_bin,
                        context_vector, tpl_features,
                        bandit_tpl, bandit_pol, normalizer,
                        live_logger, round_id, bidx, global_step):

    model.eval()
    cand = bandit_select(args, context_vector, live_logger,
                         allowed_tpl_ids, policy_space,
                         bandit_tpl, bandit_pol,
                         tpl_features, pol_features=None,
                         epoch=round_id, bidx=bidx, global_step=global_step)

    tpl_id = cand["tpl_id"]
    policy_name = cand["policy_name"]
    pol_idx = cand["pol_idx"]

    probe_loss, probe_mse, probe_mae = evaluate_metrics(
        model, tokenizer, probe_loader, templates, tpl_id, args,
        news_df, policy_name, policy_kw, device, volatility_bin=volatility_bin
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
        f"probe_loss={probe_loss:.4f} probe_mse={probe_mse:.4f} probe_mae={probe_mae:.4f} "
        f"reward_norm={r_hat:.4f}"
    )

    return tpl_id, policy_name, pol_idx


def main(args):
    filename = "log_rl_" + str(args.rl_use) + "_epoch_" + str(args.epochs) + "_" + args.taskName
    log_filename = filename + ".log"
    live_logger, live_path, log_jsonl = setup_live_logger(save_dir=args.save_dir + "/" + args.taskName, filename=log_filename)
    print(f"[live log] {live_path}  (实时查看: tail -f '{live_path}')")

    # 准备记录 prompt 的文件，清理旧内容
    ckpt_dir = os.path.join("./checkpoints", args.taskName)
    os.makedirs(ckpt_dir, exist_ok=True)
    prompt_path = os.path.join(ckpt_dir, f"prompts_{args.taskName}.json")
    with open(prompt_path, "w", encoding="utf-8") as f:
        pass

    # 准备记录 prompt 的文件，清理旧内容
    ckpt_dir = os.path.join("./checkpoints", args.taskName)
    os.makedirs(ckpt_dir, exist_ok=True)
    ans_jsonl_path = os.path.join(ckpt_dir, f"test_answers_{args.taskName}.jsonl")
    with open(ans_jsonl_path, "w", encoding="utf-8") as f:
        pass

    # 每次运行先清空 true_pred csv,写表头
    test_value_dir = os.path.join("./checkpoints", args.taskName)
    
    true_pred_csv_path = os.path.join(test_value_dir, f"true_pred_{args.taskName}.csv")
    os.makedirs(os.path.dirname(true_pred_csv_path), exist_ok=True)
    with open(true_pred_csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["pred", "true"])
    
    set_seed(args.seed)
    device = device_from_id(args.gpu)

    def _read(path):
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    train_df = _read(args.train_file)
    val_df = _read(args.val_file)
    test_df = _read(args.test_file)

    train_df[args.time_col] = pd.to_datetime(train_df[args.time_col], dayfirst=args.dayFirst)
    val_df[args.time_col] = pd.to_datetime(val_df[args.time_col], dayfirst=args.dayFirst)
    test_df[args.time_col] = pd.to_datetime(test_df[args.time_col], dayfirst=args.dayFirst)

    train_loader = make_loader(train_df, args.time_col, args.value_col,
                               args.history_len, args.horizon, args.stride, args.batch_size,
                               shuffle=True, id_col=args.id_col, dayFirst=args.dayFirst)
    val_loader = make_loader(val_df, args.time_col, args.value_col,
                             args.history_len, args.horizon, args.stride, args.batch_size,
                             shuffle=False, id_col=args.id_col, dayFirst=args.dayFirst)
    test_loader = make_loader(test_df, args.time_col, args.value_col,
                              args.history_len, args.horizon, args.stride, args.batch_size,
                              shuffle=False, id_col=args.id_col, dayFirst=args.dayFirst)

    news_df = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])
    news_df[args.news_time_col] = pd.to_datetime(news_df[args.news_time_col], dayfirst=args.dayFirst)
    if args.news_path:
        news_df = load_news(args.news_path, args.news_time_col, args.news_tz)

    policy_kw = _load_keywords(args.keyword_path)
    templates = load_templates(args.template_pool)

    tokenizer, model = load_llama_lora(
        base_model=args.base_model,
        tokenizer_id=args.tokenizer,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.target_modules,
        load_in_4bit=args.load_in_4bit, gradient_checkpointing=args.gradient_checkpointing,
        max_seq_len=args.max_seq_len, device=device, horizon=args.horizon
    )
    model.to(device)

    optim = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
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
        num_training_steps=total_opt_steps
    )

    volatility_bin = compute_volatility_bin(train_df, time_col=args.time_col, value_col=args.value_col,
                                            window=args.history_len, bins=args.volatility_bin_tiers, dayfirst=args.dayFirst)
    print(f"Computed volatility_bin for training set = {volatility_bin}")
    volatility_bin_val = compute_volatility_bin(val_df, time_col=args.time_col, value_col=args.value_col,
                                                window=args.history_len, bins=args.volatility_bin_tiers, dayfirst=args.dayFirst)
    print(f"Computed volatility_bin for validation set = {volatility_bin_val}")
    volatility_bin_test = compute_volatility_bin(test_df, time_col=args.time_col, value_col=args.value_col,
                                                 window=args.history_len, bins=args.volatility_bin_tiers, dayfirst=args.dayFirst)
    print(f"Computed volatility_bin for testing set = {volatility_bin_test}")

    live_logger.info(f"-----------------------------------------------------")
    live_logger.info(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    live_logger.info(f"Training started with volatility bins: trainset={volatility_bin}, valset={volatility_bin_val}, testset={volatility_bin_test}")
    live_logger.info(f"Volatility bin tiers: {args.volatility_bin_tiers}")
    live_logger.info(f"-----------------------------------------------------")
    live_logger.info(f"RL settings: rl_use={args.rl_use}, rl_algo={args.rl_algo}, reward_metric={args.reward_metric}, reward_mode={args.reward_mode}, "
                     f"select_policy_by={args.select_policy_by}, rl_cycle_steps={args.rl_cycle_steps}, rl_update_times={args.rl_update_times}")
    live_logger.info(f"-----------------------------------------------------")
    live_logger.info(f"Templates loaded: {len(templates)} templates from {args.template_pool}")
    live_logger.info(f"news_topK={args.news_topK}, news_topM={args.news_topM}, news_window_days={args.news_window_days}")
    live_logger.info(f"News retrieval policy keywords loaded from: {args.keyword_path}, total number of policy keywords: {len(policy_kw)}")
    live_logger.info(f"Epochs: {args.epochs}, Max Steps: {args.max_steps}, Early Stop Patience: {args.early_stop_patience}")
    live_logger.info(f"Base model: {args.base_model}, LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, target_modules={args.target_modules}")
    live_logger.info(f"Max seq len: {args.max_seq_len}, History len: {args.history_len}, Horizon: {args.horizon}, Stride: {args.stride}")
    live_logger.info(f"Description: {args.description}")
    live_logger.info(f"news_dropout={args.news_dropout}")
    live_logger.info(f"target_precision={args.target_precision}")
    # live_logger.info(f"eval_gen_batches={int(getattr(args, 'eval_gen_batches', 5))}, eval_gen_per_batch={int(getattr(args, 'eval_gen_per_batch', 1))}, gen_max_new_tokens={int(getattr(args, 'gen_max_new_tokens', 256))}")
    live_logger.info(f"-----------------------------------------------------")
    live_logger.info(f"Device: {device}, Model dtype: {next(model.parameters()).dtype}")
    live_logger.info(f"Optimizer: AdamW, LR: {args.lr}, Weight Decay: {args.weight_decay}")
    live_logger.info(f"Scheduler: Cosine with Warmup, Total Steps: {total_opt_steps}, Warmup Steps: {warmup_steps}")
    # effective batch size: 一次更新等效看到了多少样本
    # 每次累积用 batch_size 条
    # 累积 grad_accum 次
    # 一次更新总共用到：batch_size * grad_accum 条样本
    live_logger.info(f"Batch size: {args.batch_size}, Gradient Accumulation: {args.grad_accum}, Effective Batch Size: {args.batch_size * args.grad_accum}")
    live_logger.info(f"Token budget: {args.token_budget} (history frac: {args.token_budget_history_frac}, news frac: {args.token_budget_news_frac})")
    live_logger.info(f"-----------------------------------------------------")

    normalizer = RewardNormalizer(ema=args.reward_ema, use_group_norm=args.domain_reward_norm)
    val_state = ValidationState(ema_alpha=args.val_ema_alpha)

    context_vector = encode_instruction(args, ctx={}, volatility_bin=volatility_bin)

    tpl_features, feat_dim = make_tpl_feature_fn(
        templates=templates,
        add_one_hot=True,
        add_cost_proxy=False,
        add_cross_terms=True,
    )
    allowed_tpl_ids = sorted([t['id'] for t in templates.values()])

    d_tpl = len(context_vector) + len(tpl_features(allowed_tpl_ids[0], context_vector=context_vector))
    d_pol = len(context_vector)
    bandit_tpl = LinTS(d_tpl, v=args.ts_v) if args.rl_algo == 'lints' else LinUCB(d_tpl, alpha=args.ucb_alpha)
    policy_space = ['keywords', 'sentiment', "keyword_sentiment_hybrid"]
    bandit_pol = LinTS(d_pol, v=args.ts_v) if args.rl_algo == 'lints' else LinUCB(d_pol, alpha=args.ucb_alpha)

    global_step = 0
    best_metric = float('inf')
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
        xlabel = "Eval period"

        plt.figure()
        plt.plot(epochs, val_loss_per_epoch, label="Val Loss")
        plt.xlabel(xlabel)
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(p, f"ValLoss_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        live_logger.info(f"Saved loss curve to {fig_path}")

        plt.figure()
        plt.plot(epochs, mse_loss_per_epoch, label="Val MSE")
        plt.xlabel(xlabel)
        plt.ylabel("MSE")
        plt.title("Validation MSE")
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(p, f"ValMSE_{args.taskName}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        live_logger.info(f"Saved validation MSE curve to {fig_path}")

        plt.figure()
        plt.plot(epochs, mae_loss_per_epoch, label="Val MAE")
        plt.xlabel(xlabel)
        plt.ylabel("MAE")
        plt.title("Validation MAE")
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
            #折线图,分别画出true和pred
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
            live_logger.info(f"Saved Pred vs True scatter plot to {fig_path}")
        except Exception as e:
            live_logger.error(f"Failed to draw Pred vs True plot: {e}")
        
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        if (args.select_policy_by == "epoch") and args.rl_use == 1:
            context_vector = get_context_features(
                None, news_df, args,
                prev_model_loss_n=None, prev_model_loss_ema_n=None,
                val_state=val_state, train_loader=train_loader,
                volatility_bin=volatility_bin
            )

            tpl_id, policy_name, pol_idx = bandit_round_update(
                model=model, tokenizer=tokenizer, probe_loader=val_loader,
                templates=templates, allowed_tpl_ids=allowed_tpl_ids,
                news_df=news_df, policy_space=policy_space, policy_kw=policy_kw,
                args=args, device=device, volatility_bin=volatility_bin_val,
                context_vector=context_vector, tpl_features=tpl_features,
                bandit_tpl=bandit_tpl, bandit_pol=bandit_pol,
                normalizer=normalizer, live_logger=live_logger,
                round_id=epoch + 1, bidx=None, global_step=global_step
            )

            live_logger.info(
                f"EPOCH_BEGIN epoch={epoch+1}, selected_template_id={tpl_id}, selected_policy={policy_name}"
            )

        for bidx, batch in enumerate(pbar):
            if (args.select_policy_by == "batch") and args.rl_use == 1:
                context_vector = get_context_features(
                    batch, news_df, args,
                    prev_model_loss_n=None, prev_model_loss_ema_n=None,
                    val_state=val_state, train_loader=train_loader,
                    volatility_bin=volatility_bin
                )

                tpl_id, policy_name, pol_idx = bandit_round_update(
                    model=model, tokenizer=tokenizer, probe_loader=val_loader,
                    templates=templates, allowed_tpl_ids=allowed_tpl_ids,
                    news_df=news_df, policy_space=policy_space, policy_kw=policy_kw,
                    args=args, device=device, volatility_bin=volatility_bin_val,
                    context_vector=context_vector, tpl_features=tpl_features,
                    bandit_tpl=bandit_tpl, bandit_pol=bandit_pol,
                    normalizer=normalizer, live_logger=live_logger,
                    round_id=epoch * len(pbar) + bidx, bidx=bidx, global_step=global_step
                )

            input_ids, attn, labels, prompt_lens, metas = forward_batch_build_inputs(
                batch, tokenizer, templates, tpl_id, args,
                news_df, policy_name, policy_kw,
                news_encoder=None, volatility_bin=volatility_bin, epoch=epoch, record_prompt=True
            )

            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)  # Long labels with -100 mask

            for _ in range(args.rl_cycle_steps):
                model.train()
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss
                # Normalize loss by gradient accumulation steps
                loss = loss / args.grad_accum
                loss.backward()
                

                log_interval = 10
                loss_window.append(float(loss.detach().cpu()))
                if global_step % log_interval == 0:
                    avg_train_loss = sum(loss_window) / len(loss_window)
                    pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}")

                if (global_step + 1) % args.grad_accum == 0:
                    optim.step()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)

                # # 定期把“真实生成的 pred vs true”写入 csv（不要每步写，会很慢）
                # if write_pred_every > 0 and (global_step % write_pred_every == 0):
                #     try:
                #         i = 0  # 只写 batch 第一个样本
                #         pl = int(prompt_lens[i].item())
                #         prompt_ids = input_ids[i:i+1, :pl]
                #         prompt_attn = attn[i:i+1, :pl]

                #         pred_vals, pred_txt = _generate_pred_numbers(
                #             model, tokenizer, prompt_ids, prompt_attn,
                #             H=args.horizon, device=device, max_new_tokens=gen_max_new_tokens
                #         )

                #         if pred_vals is not None:
                #             true_vals = batch["target_value"][i].detach().cpu().numpy().reshape(-1).tolist()
                #             pred_flat = pred_vals[:args.horizon]
                #             true_flat = [float(x) for x in true_vals[:args.horizon]]

                #             need_header = (not os.path.exists(true_pred_csv_path)) or (os.path.getsize(true_pred_csv_path) == 0)
                #             with open(true_pred_csv_path, "a", newline="") as f:
                #                 writer = csv.writer(f)
                #                 if need_header:
                #                     writer.writerow(["pred", "true"])
                #                 writer.writerows(zip(pred_flat, true_flat))
                #         else:
                #             live_logger.warning(f"[gen] parse failed at step={global_step}. gen_text={_short(pred_txt, 240)}")
                #     except Exception as e:
                #         live_logger.error(f"[gen->csv] failed at step={global_step}: {e}")

                global_step += 1

                # if (global_step % 100 == 0):
                    # val_model_loss, val_mse, val_mae = evaluate_metrics(
                    #     model, tokenizer, val_loader, templates, tpl_id, args,
                    #     news_df, policy_name, policy_kw, device,
                    #     volatility_bin=volatility_bin_val
                    # )

                    # pbar.set_postfix({
                    #     "val_model_loss": f"{val_model_loss:.4f}",
                    #     "val_mse": f"{val_mse:.4f}",
                    #     "val_mae": f"{val_mae:.4f}",
                    #     "reward_metric": f"{args.reward_metric}"
                    # })

                    # if args.reward_metric == "loss":
                    #     metric_now = val_model_loss
                    # elif args.reward_metric == "mse":
                    #     metric_now = val_mse
                    # else:
                    #     metric_now = val_mae

                    # if metric_now < best_metric - 1e-4:
                    #     best_metric = metric_now

                    # live_logger.info(
                    #     f"EVAL epoch={epoch+1} batch={bidx} step={global_step} "
                    #     f"tpl_id={tpl_id} policy={policy_name} "
                    #     f"val_model_loss={val_model_loss:.4f} val_mse={val_mse:.4f} val_mae={val_mae:.4f} "
                    #     f"best={best_metric:.4f}"
                    # )

                    # val_loss_per_epoch.append(val_model_loss)
                    # mse_loss_per_epoch.append(val_mse)
                    # mae_loss_per_epoch.append(val_mae)

                    # if args.reward_metric == "loss":
                    #     val_state.update(val_model_loss)
                    # elif args.reward_metric == "mse":
                    #     val_state.update(val_mse)
                    # else:
                    #     val_state.update(val_mae)
        
        # live_logger.info("Epoch completed. Starting evaluation on validation set...")
        # val_model_loss, val_mse, val_mae = evaluate_metrics(
        #     model, tokenizer, val_loader, templates, tpl_id, args,
        #     news_df, policy_name, policy_kw, device,
        #     volatility_bin=volatility_bin_val
        # )

        # pbar.set_postfix({
        #     "val_model_loss": f"{val_model_loss:.4f}",
        #     "val_mse": f"{val_mse:.4f}",
        #     "val_mae": f"{val_mae:.4f}",
        #     "reward_metric": f"{args.reward_metric}"
        # })

        # if args.reward_metric == "loss":
        #     metric_now = val_model_loss
        # elif args.reward_metric == "mse":
        #     metric_now = val_mse
        # else:
        #     metric_now = val_mae

        # if metric_now < best_metric - 1e-4:
        #     best_metric = metric_now

        # live_logger.info(
        #     f"EVAL epoch={epoch+1} step={global_step} "
        #     f"tpl_id={tpl_id} policy={policy_name} "
        #     f"val_model_loss={val_model_loss:.4f} val_mse={val_mse:.4f} val_mae={val_mae:.4f} "
        # )

        # val_loss_per_epoch.append(val_model_loss)
        # mse_loss_per_epoch.append(val_mse)
        # mae_loss_per_epoch.append(val_mae)

        # if args.reward_metric == "loss":
        #     val_state.update(val_model_loss)
        # elif args.reward_metric == "mse":
        #     val_state.update(val_mse)
        # else:
        #     val_state.update(val_mae)


        # Early stop (based on val_loss_per_epoch)
        if len(val_loss_per_epoch) >= 2:
            best_so_far = min(val_loss_per_epoch[:-1])
            if val_loss_per_epoch[-1] < (best_so_far - 1e-4):
                stale_rounds = 0
            else:
                args.early_stop_patience = 3
                stale_rounds += 1
                print(f"[Early Stop] {stale_rounds} out of {args.early_stop_patience}")

                if stale_rounds >= args.early_stop_patience:
                    print("Early stopping triggered.")
                    live_logger.info(f"Early stopping triggered at epoch {epoch+1} and batch {bidx}.")
                    draw_metric_trend()

                    if test_loader is not None:
                        test_model_loss, test_mse, test_mae = evaluate_metrics(
                            model, tokenizer, test_loader, templates, tpl_id, args,
                            news_df, policy_name, policy_kw, device, volatility_bin=volatility_bin_test, testing=True, true_pred_csv_path=true_pred_csv_path
                        )
                        tqdm.write(f"[TEST] model_loss = {test_model_loss:.4f} mse={test_mse:.4f}  mae={test_mae:.4f}")
                        live_logger.info(f"[TEST] model_loss = {test_model_loss:.4f} mse={test_mse:.4f}  mae={test_mae:.4f}")
                        record_test_results_csv(test_mse, test_mae)
                        draw_pred_true()
                    return

    # draw_metric_trend()
    

    if test_loader is not None:
        test_model_loss, test_mse, test_mae = evaluate_metrics(
            model, tokenizer, test_loader, templates, tpl_id, args,
            news_df, policy_name, policy_kw, device, volatility_bin=volatility_bin_test, testing=True, true_pred_csv_path=true_pred_csv_path
        )
        live_logger.info(f"-----------------------------------------------------")
        tqdm.write(f"[TEST] model_loss = {test_model_loss:.4f} mse={test_mse:.4f}  mae={test_mae:.4f}")
        live_logger.info(f"[TEST] model_loss = {test_model_loss:.4f} mse={test_mse:.4f}  mae={test_mae:.4f}")
        record_test_results_csv(test_mse, test_mae)
        draw_pred_true()
