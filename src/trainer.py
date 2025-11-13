import os, json
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Subset
from tqdm import tqdm
from itertools import islice
from .utils.utils import set_seed, device_from_id
from .data_construction.data import make_loader
from .news_rules import load_news, get_candidates, select_news, _load_keywords, get_num_news_between
from .data_construction.prompt import load_templates, format_history, format_news, build_prompt
from .RL.rl_bandit import LinTS, LinUCB, RewardNormalizer
from .utils.metrics import rmse, mae, smape
from .model import load_llama_lora, SPECIAL_PRED_TOKEN
from collections import deque
from .news_rules import NewsEncoder
from .ValidationState import ValidationState
import logging, os
from logging.handlers import WatchedFileHandler
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_cosine_schedule_with_warmup
import math
from .utils.logger import setup_live_logger
from .data_construction.probation import prepare_val_probe_by_frac, prepare_val_probe_by_number
from .RL.features import bandit_select, compute_time_series_context_from_batch, compute_news_density_context, encode_instruction, get_context_features

# METRIC_FN = {'rmse': rmse, 'mae': mae, 'smape': smape}

def _short(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"

# 把一个 batch 的结构化样本 → prompt 文本 → tokenizer → 张量
# 这里的 batch 是 SlidingDataset 的输出格式
def forward_batch_build_inputs(batch, tokenizer, templates, tpl_id, args,
                               news_df, policy_name, policy_kw, news_encoder, volatility_bin):
    # history_len, pred_len
    L, H = args.history_len, args.horizon
    # token budget
    hist_budget = int(args.token_budget * args.token_budget_history_frac)
    news_budget = int(args.token_budget * args.token_budget_news_frac)
    # prompt模板
    tpl_text = templates[tpl_id]['text']
    prompts, targets = [], []
    metas = []
    # print(len(batch['history']), len(batch['target']), len(batch['target_time']))

    for i in range(len(batch['history'])):      
        history = batch['history'][i].tolist()
        target = batch['target'][i].tolist()
        t_target = batch['target_time'][i]
        series_id = batch['series_id'][i] if isinstance(batch['series_id'], list) else batch['series_id']

        # 对齐 target_time 到新闻的时区
        cand = get_candidates(news_df, args.news_time_col, t_target, args.news_window_days, args.news_topM)

        # 构造 query_text（任务上下文 + 区域等；可按需改）
        query_text = f"region={args.region} unit={args.unit} freq={args.freq_min}m history_len={args.history_len} horizon={H} "\
                    f"volatility={volatility_bin}"

        # now_ts 使用目标时间 t_target；确保 cand_df 的时间列已是 datetime64
        # now_ts = pd.to_datetime(t_target, errors="coerce")
        now_ts = t_target

        selected = select_news(
            cand, policy_name, args.news_text_col, policy_kw, args.news_topK
        )
        hist_str = format_history(history, args.unit, hist_budget, tokenizer)
        news_str = format_news(selected, args.news_text_col, news_budget, tokenizer,
                               summary_method=args.news_summary_method, max_sentences=args.news_max_sentences)
        
        start_date=batch['history_times'][0][i]
        end_date=batch['history_times'][-1][i]

        prediction_start =batch['target_times'][0][i]
        prediction_end =batch['target_times'][-1][i]
        # Build prompt
        prompt = build_prompt(tpl_text, L, H, args.unit, args.description, hist_str, news_str,
                              start_date=start_date,
                              end_date=end_date,
                              freq=args.freq_min, value_col=args.value_col, 
                              pred_end=prediction_end, pred_start=prediction_start)
        prompt = prompt + f"\n{SPECIAL_PRED_TOKEN}\n"

        # print("-------prompt-------")
        # print(prompt)
        # print("-------target-------")
        # print(target)

        prompts.append(prompt)
        targets.append(target)
        # 收集本样本的“被选新闻”元信息（尽量简短）
        sel_info = []
        if len(selected) > 0:
            for _, r in selected.iterrows():
                sel_info.append({
                    "t": (pd.to_datetime(r[args.news_time_col], errors="coerce").isoformat()
                          if pd.notna(r[args.news_time_col]) else None),
                    "title": r.get("title", None),
                    "text_snippet": (str(r.get(args.news_text_col, ""))[:80] if args.news_text_col in r else None)
                })

        metas.append({
            "series_id": series_id,
            "target_time": pd.to_datetime(t_target, errors="coerce").isoformat() if pd.notna(t_target) else None,
            "selected_news": sel_info,
            "Candidates_found": len(cand),
        })

    enc = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors='pt')
    input_ids = enc['input_ids']
    # print("Input IDs shape:", input_ids.shape)
    attn = enc['attention_mask']
    labels = torch.tensor(targets, dtype=torch.float32)

    # 计算每个样本的 token 长度，用于日志
    tok_lens = attn.sum(dim=1).tolist()
    for i in range(len(metas)):
        metas[i]["tok_len"] = int(tok_lens[i])


    return input_ids, attn, labels, metas


def evaluate_metrics(model, tokenizer, data_loader, templates, tpl_id, args,
                         news_df, policy_name, policy_kw, device, volatility_bin):
    model.eval()
    loss_sum, n_samples = 0.0, 0
    mse_sum, mae_sum, n_elems = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in data_loader:
            # 构造输入
            input_ids, attn, labels, metas = forward_batch_build_inputs(
                batch, tokenizer, templates, tpl_id, args,
                news_df, policy_name, policy_kw,
                news_encoder=None, volatility_bin=volatility_bin
            )
            input_ids = input_ids.to(device)
            attn      = attn.to(device)

            # 模型的 dtype（bfloat16/float16/float32）
            model_dtype = next(model.parameters()).dtype
            labels_for_loss = labels.to(device=device, dtype=model_dtype)

            # 一次前向，同时拿 loss 和 pred
            out = model(input_ids=input_ids, attention_mask=attn,
                        labels=labels_for_loss)

            # 1) 统计 loss（按样本数平均）
            batch_loss = out["loss"]
            # 假设 loss 是“batch 内平均”，乘上样本数得到总和
            loss_sum += float(batch_loss.detach().cpu()) * labels.size(0)
            n_samples += labels.size(0)

            # 2) 统计 MSE/MAE（按元素数平均）
            # 用 fp32 算数值更稳
            y_hat = out["pred"].to(dtype=torch.float32, device=device)
            labels_fp32 = labels.to(device=device, dtype=torch.float32)

            mse = torch.nn.functional.mse_loss(y_hat, labels_fp32, reduction="sum")
            mae = torch.nn.functional.l1_loss(y_hat, labels_fp32, reduction="sum")

            mse_sum += float(mse.detach().cpu())
            mae_sum += float(mae.detach().cpu())
            n_elems += labels_fp32.numel()

    loss_avg = loss_sum / max(1, n_samples)
    mse_avg  = mse_sum  / max(1, n_elems)
    mae_avg  = mae_sum  / max(1, n_elems)

    return loss_avg, mse_avg, mae_avg

def compute_volatility_bin(df, time_col="", value_col="", 
                           window=48, bins = 10, dayfirst=True):
    """
    自动计算电价数据集的 volatility_bin
    
    参数:
      df        : pandas.DataFrame，包含时间列和电价列
      time_col  : 时间列名
      value_col : 电价列名
      window    : rolling 窗口大小 (默认 48 = 2 天, 如果是半小时频率)
      bins      : 分档数 (默认 5 档, 0=低波动, 4=高波动)
      dayfirst  : 时间解析是否日优先

    返回:
      volatility_bin (int): 档位编号 [0, bins-1]
    """
    # 确保时间列为 datetime
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=dayfirst)
    df = df.sort_values(time_col)

    # 取最近 window 个点的 rolling std
    recent = df[value_col].iloc[-window:]
    if len(recent) < 2:
        return 0  # 数据太少，默认最低档
    
    vol = recent.std()

    # 用历史分布确定档位
    all_std = df[value_col].rolling(window).std().dropna()
    if len(all_std) == 0:
        return 0
    
    # 分位数作为阈值
    thresholds = np.quantile(all_std, np.linspace(0, 1, bins+1)[1:-1])
    bin_id = np.digitize(vol, thresholds, right=True)
    return int(min(bin_id, bins-1))


def make_tpl_feature_fn(templates,
                        add_one_hot=True,
                        add_cost_proxy=False,
                        add_cross_terms=False):
    """
    返回：
      tpl_features(tid, context_vector=None) -> np.ndarray(float32)
      feat_dim(context_dim=None) -> int   # 计算拼接后的维度，供 bandit 初始化
    """
    T = len(templates)
    # print(f"[Bandit] 模板数={T}， template: {templates}")
    # 统计归一化所需的范围
    n_paths_list = [float(t.get('n_paths', 1) or 1) for t in templates]
    max_n_paths = max(n_paths_list) if n_paths_list else 1.0

    # breath_intensity = has_breath * (1 / breath_freq)
    raw_breath_intensity = []
    for t in templates:
        hb = float(bool(t.get('has_breath', False)))
        bf = float(t.get('breath_freq', 0) or 0)
        intensity = hb * (1.0 / bf) if (hb > 0 and bf > 0) else 0.0
        raw_breath_intensity.append(intensity)
    bi_min = min(raw_breath_intensity) if raw_breath_intensity else 0.0
    bi_max = max(raw_breath_intensity) if raw_breath_intensity else 1.0
    bi_range = (bi_max - bi_min) if (bi_max - bi_min) > 0 else 1.0

    # 复杂度/成本 proxy（越大代表预计 token/时延越高；只做相对量）
    def _cost_proxy(t):
        he  = float(bool(t.get('has_example', False)))
        hb  = float(bool(t.get('has_breath', False)))
        hd  = float(bool(t.get('has_decomp', False)))
        hsc = float(bool(t.get('has_self_consistency', False)))
        np_norm = float(t.get('n_paths', 1) or 1) / max_n_paths
        # 可按经验权重：自一致与多路径最“贵”，示例与分解次之，呼吸略增开销
        cost = (0.4*he + 0.5*hd + 1.0*hsc + 0.6*np_norm + 0.2*hb)
        return cost

    raw_costs = [_cost_proxy(t) for t in templates]
    c_min = min(raw_costs) if raw_costs else 0.0
    c_max = max(raw_costs) if raw_costs else 1.0
    c_range = (c_max - c_min) if (c_max - c_min) > 0 else 1.0

    # 预生成 one-hot
    I = np.eye(T, dtype=np.float32)

    def _single_tpl_vec(tid: int) -> np.ndarray:
        # print(templates)
        t = templates[tid]
        he  = float(bool(t.get('has_example', False)))
        hb  = float(bool(t.get('has_breath', False)))
        hd  = float(bool(t.get('has_decomp', False)))
        hsc = float(bool(t.get('has_self_consistency', False)))
        np_norm = float(t.get('n_paths', 1) or 1) / max_n_paths

        # 归一化后的呼吸强度
        bf = float(t.get('breath_freq', 0) or 0)
        bi = hb * (1.0 / bf) if (hb > 0 and bf > 0) else 0.0
        bi_norm = (bi - bi_min) / bi_range

        vec = [1.0,         # 模板偏置（每个“臂”的截距）
               he, hb, hd, hsc,
               np_norm,
               bi_norm]

        if add_cost_proxy:
            c = (_cost_proxy(t) - c_min) / c_range
            vec.append(c)

        if add_one_hot:
            vec.extend(I[tid].tolist())

        return np.asarray(vec, dtype=np.float32)

    # 主函数：仅在需要交叉项时，需要传入 context_vector
    def tpl_features(tid: int, context_vector) -> np.ndarray:
        arm = _single_tpl_vec(tid)
        if add_cross_terms:
            if context_vector is None:
                raise ValueError("add_cross_terms=True 时必须提供 context_vector")
            cross = np.outer(context_vector.astype(np.float32), arm).ravel().astype(np.float32)
            return np.concatenate([arm, cross], axis=0).astype(np.float32)
        else:
            return arm

    # 提供一个维度计算器，便于初始化 bandit
    def feat_dim(context_dim) -> int:
        base = len(_single_tpl_vec(0))
        if add_cross_terms:
            if context_dim is None:
                raise ValueError("feat_dim 需要 context_dim 当 add_cross_terms=True")
            return base + context_dim * base
        else:
            return base

    return tpl_features, feat_dim

def main(args):
    log_filename = "log_rl_"+str(args.rl_use)+"_epoch_"+str(args.epochs)+".log"
    live_logger, live_path, log_jsonl = setup_live_logger(save_dir=args.save_dir, filename=log_filename)
    print(f"[live log] {live_path}  (实时查看: tail -f '{live_path}')")
    set_seed(args.seed)
    device = device_from_id(args.gpu)
    def _read(path):
        if path.endswith('.parquet'): return pd.read_parquet(path)
        return pd.read_csv(path)

    train_df = _read(args.train_file)
    val_df = _read(args.val_file)
    test_df = _read(args.test_file)

    train_df[args.time_col] = pd.to_datetime(train_df[args.time_col], dayfirst=args.dayFirst)
    val_df[args.time_col] = pd.to_datetime(val_df[args.time_col],dayfirst=args.dayFirst)
    test_df[args.time_col] = pd.to_datetime(test_df[args.time_col])

    train_loader = make_loader(train_df, args.time_col, args.value_col,
                               args.history_len, args.horizon, args.stride, args.batch_size, shuffle=True, id_col=args.id_col, dayFirst=args.dayFirst)
    val_loader = make_loader(val_df, args.time_col, args.value_col,
                             args.history_len, args.horizon, args.stride, args.batch_size, shuffle=True, id_col=args.id_col, dayFirst=args.dayFirst)
    test_loader = make_loader(test_df, args.time_col, args.value_col,
                            args.history_len, args.horizon, args.stride, args.batch_size,
                            shuffle=True, id_col=args.id_col, dayFirst= args.dayFirst)

    # probe_loader = prepare_val_probe_by_frac(val_loader, frac=args.rl_val_probe_frac,seed=args.seed)
        
    # ===== News retrieval setup =====
    news_df = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])
    news_df[args.news_time_col] = pd.to_datetime(news_df[args.news_time_col], dayfirst=args.dayFirst)
    if args.news_path:
        news_df = load_news(args.news_path, args.news_time_col, args.news_tz)
    # print("news_df dates:",news_df[args.news_time_col])
    
    #  ============ Build news encoder (SBERT or TF-IDF) ===========
    policy_kw = _load_keywords(args.keyword_path)
    # ===== Load Prompt templates =====
    templates = load_templates(args.template_pool)
    # 允许指定模板 ID 列表，或者使用所有模板

    tokenizer, model = load_llama_lora(
        base_model=args.base_model,
        tokenizer_id=args.tokenizer,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.target_modules,
        load_in_4bit=args.load_in_4bit, gradient_checkpointing=args.gradient_checkpointing,
        max_seq_len=args.max_seq_len, device=device, horizon=args.horizon
    )
    # model.config.pad_token_id = tokenizer.pad_token_id
    # if hasattr(model, "generation_config"):
    #     model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    optim = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    def lr_lambda(current_step):
        warmup_steps = int(0.1 * args.max_steps)   # 前10% steps做warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # warmup之后做余弦衰减
        progress = float(current_step - warmup_steps) / float(max(1, args.max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # ✅ 用 math.cos


    #HuggingFace Transformers 的工具函数
    num_batches = len(train_loader)            # 批次数
    opt_steps_per_epoch = math.ceil((num_batches * max(1, args.rl_cycle_steps)) / max(1, args.grad_accum))
    total_opt_steps = opt_steps_per_epoch * args.epochs
    warmup_steps = int(getattr(args, "warmup_ratio", 0.1) * total_opt_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps
    )
    # scheduler = LambdaLR(optim, lr_lambda)

    # 计算 volatility_bin, bins=10 means 10 tiers
    volatility_bin  = compute_volatility_bin(train_df, time_col=args.time_col, value_col=args.value_col, window=args.history_len, bins=args.volatility_bin_tiers, dayfirst=args.dayFirst)
    print(f"Computed volatility_bin for training set = {volatility_bin}")
    volatility_bin_val  = compute_volatility_bin(val_df, time_col=args.time_col, value_col=args.value_col, window=args.history_len, bins=args.volatility_bin_tiers, dayfirst=args.dayFirst)
    print(f"Computed volatility_bin for validation set = {volatility_bin_val}")
    volatility_bin_test = compute_volatility_bin(test_df, time_col=args.time_col, value_col=args.value_col,window=args.history_len, bins=args.volatility_bin_tiers, dayfirst=args.dayFirst)
    print(f"Computed volatility_bin for testing set = {volatility_bin_test}")
    
    # Normalizer for reward scaling
    normalizer = RewardNormalizer(ema=args.reward_ema, use_group_norm=args.domain_reward_norm)
    val_state = ValidationState(ema_alpha=args.val_ema_alpha)


    # ======= Bandit setup =======
    context_vector = encode_instruction(args, ctx={}, volatility_bin=volatility_bin)

    # def tpl_features(tid): return np.array([templates[tid].get('has_explain',0)], dtype=np.float32)
    # print(sorted(templates.values(), key=lambda t: t['id']))
    tpl_features,  feat_dim = make_tpl_feature_fn(
        sorted(templates.values(), key=lambda t: t['id']),
        add_one_hot=True,
        add_cost_proxy=False,
        add_cross_terms=True,
    )
    allowed_tpl_ids = sorted([t['id'] for t in templates.values()])
    
    d_tpl = len(context_vector) + len(tpl_features(allowed_tpl_ids[0], context_vector=context_vector))
    d_pol = len(context_vector)
    bandit_tpl = LinTS(d_tpl, v=args.ts_v) if args.rl_algo=='lints' else LinUCB(d_tpl, alpha=args.ucb_alpha)
    policy_space = ['keywords', 'sentiment', "keyword_sentiment_hybrid"]
    pol_features = None
    bandit_pol = LinTS(d_pol, v=args.ts_v) if args.rl_algo=='lints' else LinUCB(d_pol, alpha=args.ucb_alpha)

    prev_metric = None
    global_step = 0
    best_metric = float('inf')
    stale_rounds = 0
    loss_window = deque(maxlen=50)   # 近 50 步的滑动平均
    tpl_id = allowed_tpl_ids[0]
    policy_name = policy_space[0]

    # ======= 主训练循环：按 epoch 迭代 =======
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # prev_model_loss_n = context_vector.get("prev_model_loss_n", None)
        # prev_model_loss_ema_n = context_vector.get("prev_model_loss_ema_n", None)
        # print(args.rl_use)
        if (args.select_policy_by == "epoch") and args.rl_use == 1:
            context_vector = get_context_features(None, news_df, args, prev_model_loss_n=None, prev_model_loss_ema_n=None, val_state=val_state, train_loader=train_loader,volatility_bin=volatility_bin)
            bandit_result = bandit_select(args, context_vector, live_logger, allowed_tpl_ids, policy_space,
                          bandit_tpl, bandit_pol, tpl_features, pol_features, epoch, None, None)
            
            live_logger.info(f"EPOCH_BEGIN epoch={epoch+1}, template_id={bandit_result['tpl_id']}, policy={bandit_result['policy_name']}")    

        # ======== 训练主循环：按批次迭代 ========
        for bidx, batch in enumerate(pbar):
            # print(batch)
            if (args.select_policy_by == "batch") and args.rl_use == 1:
                context_vector = get_context_features(batch, news_df, args, prev_model_loss_n=None, prev_model_loss_ema_n=None, val_state=val_state, train_loader=train_loader,volatility_bin=volatility_bin)
                bandit_result = bandit_select(args, context_vector, live_logger, allowed_tpl_ids, policy_space,
                          bandit_tpl, bandit_pol, tpl_features, pol_features,epoch, None, None)
            
            if args.rl_use == 1:
                tpl_id = bandit_result["tpl_id"]
                policy_name = bandit_result["policy_name"]
                pol_idx = bandit_result["pol_idx"]
            

            # 用选好的“模板 + 策略”构建本批次输入
            input_ids, attn, labels, metas = forward_batch_build_inputs(
                batch, tokenizer, templates, tpl_id, args, news_df, policy_name, policy_kw, news_encoder=None, volatility_bin=volatility_bin
            )

            input_ids = input_ids.to(device)
            attn = attn.to(device)
            model_dtype = next(model.parameters()).dtype
            labels   = labels.to(device=device, dtype=model_dtype)

            # # —— 即时记录：样本级别（被选新闻等）
            # for m in metas:
            #     titles = []
            #     for item in m.get("selected_news", []):
            #         t = item.get("title") or item.get("text_snippet") or ""
            #         titles.append(_short(t, 30))
            steps_this_cycle = max(0, args.rl_cycle_steps)
            if steps_this_cycle > 0:
                for _ in range(steps_this_cycle):
                    model.train()
                    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                    loss = out['loss']
                    loss.backward()

                    # logging
                    log_interval = 10
                    loss_window.append(float(loss.detach().cpu()))
                    if global_step % log_interval == 0:
                        avg_train_loss = sum(loss_window)/len(loss_window)
                        # 不破坏 tqdm 样式，使用 tqdm.write
                        # tqdm.write(f"[step {global_step}] train_loss={avg_train_loss:.4f}")
                        # 或者实时挂到进度条右侧（可选）
                        pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}")
                    
                    # loss.backward()
                    # optim.step()
                    # scheduler.step()   # ✅ 动态更新学习率
                    # optim.zero_grad()

                    if (global_step + 1) % args.grad_accum == 0:
                        optim.step()
                        scheduler.step()   # ✅ 动态更新学习率
                        optim.zero_grad(set_to_none=True)
                    global_step += 1

                    eval_interval = math.ceil(len(train_loader) * args.rl_cycle_steps / args.rl_update_times)
                    # validation 
                    if global_step % eval_interval == 0:
                        val_model_loss, val_mse, val_mae = evaluate_metrics(
                            model, tokenizer, val_loader, templates, tpl_id, args,
                            news_df, policy_name, policy_kw, device, volatility_bin=volatility_bin_val)
                    
                        # tqdm.write(f"[step {global_step}] val_{args.reward_metric}={val_metric:.4f}  model_loss={model_loss:.6f}")
                        pbar.set_postfix({
                            f'val_model_loss': f"{val_model_loss:.4f}",
                            f'val_mse': f"{val_mse:.4f}",
                            f'val_mae': f"{val_mae:.4f}",
                            f'reward_metric': f"{args.reward_metric}"
                        })
                        # 计算奖励
                        if args.reward_metric == "loss":
                            metric_now = val_model_loss
                        elif args.reward_metric == "mse":
                            metric_now = val_mse
                        elif args.reward_metric == "mae":
                            metric_now = val_mae
                            # set reward
                        r = 0
                        r_hat = 0
                        if args.rl_use == 1:
                            if args.reward_mode == 'delta' and prev_metric is not None:
                                r = (prev_metric - metric_now)
                            else:
                                r = -metric_now
                            # length penalty
                            tok_len = input_ids.size(1)
                            # default 0 penalty
                            r -= args.reward_len_penalty * float(tok_len)
                            # penalty for k news, default 0. topK news default 5
                            r -= args.reward_k_penalty * float(args.news_topK)
                            # normalize & possibly group-wise normalize
                            r_hat = normalizer.update_and_normalize(r, group_key=(args.region, args.horizon) if args.domain_reward_norm else None)

                        live_logger.info(
                            f"EVAL  epoch={epoch+1} batch={bidx} step={global_step} "
                            f"val_model_loss: {val_model_loss:.4f} "
                            f"val_mse: {val_mse:.4f} "
                            f"val_mae: {val_mae:.4f} "
                            f"[reward_metric]: {args.reward_metric} "
                            f"reward_norm={r_hat:.4f} avg_tok_len={float(attn.sum(dim=1).float().mean().item()):.1f} "
                            f"topK={int(args.news_topK)} best={best_metric:.4f}"
                        )

                        if args.rl_use == 1:
                            # update bandits
                            x_tpl = np.concatenate([context_vector, tpl_features(tpl_id, context_vector)], axis=0)
                            # x_pol = np.concatenate([context_vector, pol_features(pol_idx)], axis=0)
                            x_pol = context_vector.astype(np.float32)

                            bandit_tpl.update(x_tpl, r_hat)
                            bandit_pol.update(x_pol, r_hat)

                            if r_hat < 0:
                                # 惩罚时，才重新选
                                bandit_result = bandit_select(args, context_vector, live_logger, allowed_tpl_ids, policy_space,
                                bandit_tpl, bandit_pol, tpl_features, pol_features, epoch, None, None)
                                tpl_id = bandit_result["tpl_id"]
                                policy_name = bandit_result["policy_name"]
                                pol_idx = bandit_result["pol_idx"]
                                live_logger.info(f"NEGATIVE REWARD      re-selecting tpl_id={tpl_id} policy={policy_name}")


                        prev_metric = metric_now
                        pbar.set_postfix({
                            f'val_model_loss': f"{val_model_loss:.4f}",
                            f'val_mse': f"{val_mse:.4f}",
                            f'val_mae': f"{val_mae:.4f}",
                            f'reward_metric': f"{args.reward_metric}"
                        })
                        # update val_state
                        if args.reward_metric == "loss":
                            val_state.update(val_model_loss)
                        elif args.reward_metric == "mse":
                            val_state.update(val_mse)
                        elif args.reward_metric == "mae":
                            val_state.update(val_mae)

                        if metric_now < (best_metric - 1e-4):
                            # Save best
                            best_metric = metric_now
                            stale_rounds = 0
                            os.makedirs(args.save_dir, exist_ok=True)
                            # torch.save({'model': model.state_dict(), 'step': global_step},
                            #            os.path.join(args.save_dir, f'best.pt'))
                        else:
                            # Early stopping
                            stale_rounds += 1
                            print(f"[Early Stop] {stale_rounds} out of {args.early_stop_patience}")
                            if stale_rounds >= args.early_stop_patience:
                                print("Early stopping triggered.")
                                if test_loader is not None:
                                    test_mse, test_mae = evaluate_metrics(model, tokenizer, test_loader, templates, tpl_id, args,
                                                                news_df, policy_name, policy_kw, device, volatility_bin=volatility_bin_test)
                                    tqdm.write(f"[TEST] mse={test_mse:.6f}  mae={test_mae:.6f}")
                                return
                        
                    
                    # Save intermediate checkpoints
                    if args.save_interval > 0 and global_step % args.save_interval == 0:
                        os.makedirs(args.save_dir, exist_ok=True)
                        # torch.save({'model': model.state_dict(), 'step': global_step},
                        #            os.path.join(args.save_dir, f'step{global_step}.pt'))   
                                  
    if test_loader is not None:
        test_mse, test_mae = evaluate_metrics(model, tokenizer, test_loader, templates, tpl_id, args,
                                    news_df, policy_name, policy_kw, device, volatility_bin=volatility_bin_test)
        tqdm.write(f"[TEST] mse={test_mse:.6f}  mae={test_mae:.6f}")
