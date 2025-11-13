import numpy as np
import torch
from itertools import islice
from .rl_bandit import LinTS
from ..news_rules import get_num_news_between

def compute_time_series_context_from_batch(batch, eps: float = 1e-8):
    """
    仅使用 batch['history'] 计算 A 类特征（不碰 target，避免泄漏）
    返回已近似归一化的字典：std_W_n, cv_W, trend_W_n, acf_lag1_n, acf_lag48_n
    """
    history = batch["history"]
    if isinstance(history, torch.Tensor):
        H = history.detach().to(torch.float32).cpu().numpy()
    else:
        H = np.asarray([h for h in history], dtype=np.float32)

    mean_W = np.mean(H, axis=1)
    std_W  = np.std(H, axis=1)
    cv_W   = std_W / (np.abs(mean_W) + eps)

    # 简易趋势：标准化时间轴的一阶线性回归斜率（取绝对值）
    t = np.arange(H.shape[1], dtype=np.float32)
    t = (t - t.mean()) / (t.std() + eps)
    slope = ((H * t).mean(axis=1) - H.mean(axis=1) * t.mean()) / (t.var() + eps)
    trend = np.abs(slope)

    def acf(x: np.ndarray, lag: int) -> float:
        x = x - x.mean()
        if len(x) <= lag:
            return 0.0
        num = float(np.dot(x[:-lag], x[lag:]))
        den = float(np.dot(x, x)) + eps
        return num / den

    acf1  = np.array([acf(h, 1)  for h in H], dtype=np.float32)
    acf48 = np.array([acf(h, 48) for h in H], dtype=np.float32)

    # 批级聚合（均值）
    mean_W_mean = float(mean_W.mean())
    std_W_mean  = float(std_W.mean())
    cv_W_mean   = float(cv_W.mean())
    trend_mean  = float(trend.mean())
    acf1_mean   = float(acf1.mean())
    acf48_mean  = float(acf48.mean())

    # 粗略归一（经验尺度，可按数据再微调）
    std_W_n   = float(np.clip(std_W_mean / (abs(mean_W_mean) * 0.5 + 1.0), 0, 1))
    trend_W_n = float(np.clip(trend_mean / 1.0, 0, 1))
    acf1_n    = float((acf1_mean + 1.0) / 2.0)      # [-1,1] → [0,1]
    acf48_n   = float((acf48_mean + 1.0) / 2.0)

    # 标准差
    # 变异系数，相对波动性
    # 趋势强度
    # 自相关系数 序列当前值和前一时刻值的相关性
    # 长期自相关系数 序列当前值和 48 时刻前值的相关性（如日周期）
    return {
        "std_W_n": std_W_n,
        "cv_W": float(np.clip(cv_W_mean, 0, 1)),
        "trend_W_n": trend_W_n,
        "acf_lag1_n": acf1_n,
        # "acf_lag48_n": acf48_n,
    }

def compute_news_density_context(args, now_ts, news_df, density_limit = 50):
    count = get_num_news_between(news_df, args.news_time_col, now_ts, args.news_window_days)

    density_per_day = count / float(max(1, args.news_window_days))

    # 经验上界 50 条/天，超界截断；如你的数据很稠密，可把 50 调高
    return {"news_density_n": float(np.clip(density_per_day / density_limit, 0.0, 1.0))}


#把“任务/场景说明”编码成一个数值向量（比如 freq_min, horizon, token_budget, volatility_bin, need_explain, need_ci）。
# 这是上下文特征，供 Bandit 使用（即“在什么任务条件下，哪种选择更好”）。
def encode_instruction(args, ctx, volatility_bin):
    """
    将任务/场景（args）与可选的动态上下文（ctx）编码成 bandit 的上下文向量。
    ctx 可包含：A(时序统计)、E(训练态)、新闻密度等，均应已归一到[0,1]或近似范围。
    """
    features = []

    # —— 静态任务特征（归一化到相近尺度）——
    features += [                        # 波动档位，已归一到[0,1]
        args.freq_min / 60.0,                      # 频率（分钟）/ 60 → [0,1]（假设不超过1小时）
        args.horizon / 48,                     # 预测长度 / 48 → [0,1]（假设不超过48个点）
        min(args.token_budget, 2048) / 2048.0,         # 截断到 2048 再归一
        float(volatility_bin) / args.volatility_bin_tiers,             # 你是10档 → 归一到[0,1]
        1.0 if args.need_explain else 0.0,
        1.0 if args.need_ci else 0.0,
    ]

    # —— 动态上下文特征 ——（存在则添加）
    # if ctx:
    def clip01(x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    for key in [
        "std_W_n", "cv_W", "trend_W_n",
        "acf_lag1_n", "acf_lag48_n",          # A: 时序统计
        "news_density_n",                     # 新闻密度
        "prev_model_loss_n", "prev_model_loss_ema_n",  # E: 训练态
        # 若你之后也加入 delta，可在此扩展 "delta_val_n"
    ]:
        v = ctx.get(key, 0.0) #if ctx else 0.0
        features.append(float(np.clip(v, 0.0, 1.0)))  # 保守截断

    return np.array(features, dtype=np.float32)

#ε-贪心选择。用在“分数接近/想增加探索”时做随机探索
def choose_arm(scores, epsilon=0.05):
    if np.random.rand() < epsilon:
        return int(np.random.randint(len(scores)))
    return int(np.argmax(scores))

def get_context_features(batch, news_df, args, prev_model_loss_n, prev_model_loss_ema_n, val_state, train_loader, volatility_bin):

    if args.select_policy_by == "epoch":
        num_batches_in_epoch = len(train_loader)
        sample_batches_for_epoch_ctx = max(1, min(8, num_batches_in_epoch // 10))  # 10%的批次，至少1，最多8

        time_series_ctx_list = []
        news_density_ctx_list = []

        # 采样前若干 batch 粗估上下文（也可用 probe_loader 或随机采样）
        with torch.no_grad():
            for sampled_batch in islice(iter(train_loader), sample_batches_for_epoch_ctx):
                ts_ctx = compute_time_series_context_from_batch(sampled_batch)
                time_series_ctx_list.append(ts_ctx)

                # 取该批第一条样本的预测起点作为 now_ts（按你的实际结构取）
                now_ts = sampled_batch["target_times"][0][0]
                news_ctx = compute_news_density_context(
                    args,
                    now_ts=now_ts,
                    news_df=news_df,
                )
                news_density_ctx_list.append(news_ctx)

        # 聚合为 epoch 级上下文（取均值）
        def average_context(dicts):
            if not dicts:
                return {}
            keys = dicts[0].keys()
            return {k: float(np.mean([d[k] for d in dicts if k in d])) for k in keys}

        epoch_time_series_ctx = average_context(time_series_ctx_list)
        epoch_news_density_ctx = average_context(news_density_ctx_list)
        epoch_training_state_ctx = val_state.as_context()

        ctx = {**epoch_time_series_ctx, **epoch_news_density_ctx, **epoch_training_state_ctx}
        context_vector = encode_instruction(args, ctx=ctx, volatility_bin=volatility_bin)

    elif args.select_policy_by == "batch":
        # batch 级：每个 batch 动态计算
        time_series_ctx = compute_time_series_context_from_batch(batch)
        now_ts = batch["target_times"][0][0]
        news_density_ctx = compute_news_density_context(
            args,
            now_ts=now_ts,
            news_df=news_df,
        )
        training_state_ctx = val_state.as_context()
        dynamic_ctx = {**time_series_ctx, **news_density_ctx, **training_state_ctx}
        context_vector = encode_instruction(args, ctx=dynamic_ctx, volatility_bin=volatility_bin)

    return context_vector


def bandit_select(args, context_vector, live_logger, allowed_tpl_ids, policy_space,
                  bandit_tpl, bandit_pol, tpl_features, pol_features,epoch, bidx, global_step):

    if args.select_policy_by == "epoch":
    # === (A) 只在 epoch 开头选择一次 ===
        # 模板
        scores_tpl = []
        for tid in allowed_tpl_ids:
            # print(instr_feat.shape, tpl_features(tid).shape)
            # print(instr_feat)
            # print(tid)
            x = np.concatenate([context_vector, tpl_features(tid = int(tid), context_vector = context_vector)], axis=0)
            s = bandit_tpl.sample_score(x) if isinstance(bandit_tpl, LinTS) else bandit_tpl.ucb_score(x)
            scores_tpl.append(s)
        tpl_idx = choose_arm(scores_tpl, epsilon=args.epsilon)
        tpl_id  = allowed_tpl_ids[tpl_idx]

        # 新闻策略
        scores_pol = []
        for i, pol in enumerate(policy_space):
            # x = np.concatenate([context_vector, pol_features(i)], axis=0)
            x = context_vector.astype(np.float32)
            s = bandit_pol.sample_score(x) if isinstance(bandit_pol, LinTS) else bandit_pol.ucb_score(x)
            scores_pol.append(s)
        pol_idx = choose_arm(scores_pol, epsilon=args.epsilon)
        policy_name = policy_space[pol_idx]

        live_logger.info(
            f"DECISION_EPOCH_BEGIN epoch={epoch+1} "
            f"sel_template={tpl_id} policy={policy_name} "
            f"tpl_scores={[round(float(s),4) for s in scores_tpl]} "
            f"pol_scores={[round(float(s),4) for s in scores_pol]}"
        )
    # ===============================================
    elif args.select_policy_by == "batch" :
    # ======= 每个Batch都决策：选择本批次使用的“模板 + 新闻策略” =======
        # 为“模板选择”算分
        scores_tpl = []
        for tid in allowed_tpl_ids:
            # 将 “任务/场景说明” instr_feat 和 “模板特征” tpl_features 拼接成完整的上下文特征向量
            x = np.concatenate([context_vector, tpl_features(tid = int(tid),context_vector = context_vector)], axis=0)
            # 计算该模板的分数（LinTS 或 LinUCB）
            s = bandit_tpl.sample_score(x) if isinstance(bandit_tpl, LinTS) else bandit_tpl.ucb_score(x)
            scores_tpl.append(s)
        # 用 ε-贪心选择模板。
        tpl_idx = choose_arm(scores_tpl, epsilon=args.epsilon)
        tpl_id = allowed_tpl_ids[tpl_idx]

        # 为“新闻策略选择”算分
        scores_pol = []
        for i, pol in enumerate(policy_space):
            # x = np.concatenate([context_vector, pol_features(i)], axis=0)
            x = context_vector.astype(np.float32)
            s = bandit_pol.sample_score(x) if isinstance(bandit_pol, LinTS) else bandit_pol.ucb_score(x)
            scores_pol.append(s)
        # 用 ε-贪心选择模板。
        pol_idx = choose_arm(scores_pol, epsilon=args.epsilon)
        policy_name = policy_space[pol_idx]
        
        # —— 即时记录：本次“决策”层面的信息（不含样本级新闻）

        live_logger.info(
                    f"DECISION_BATCH_BEGIN epoch={epoch+1} batch={bidx} step={global_step} "
                    f"sel_template={tpl_id} policy={policy_name} "
                    f"tpl_scores={[round(float(s),4) for s in scores_tpl]} "
                    f"pol_scores={[round(float(s),4) for s in scores_pol]}"
                )
    # ===============================================
    return {"tpl_id": tpl_id, "policy_name": policy_name, "pol_idx": pol_idx}