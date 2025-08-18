import os, json
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Subset
from tqdm import tqdm

from .utils import set_seed, device_from_id
from .data import make_loader
from .news_rules import load_news, get_candidates, select_news, _load_keywords
from .prompt import load_templates, format_history, format_news, build_prompt
from .rl_bandit import LinTS, LinUCB, RewardNormalizer
from .metrics import rmse, mae, smape
from .model import load_llama_lora, SPECIAL_PRED_TOKEN

METRIC_FN = {'rmse': rmse, 'mae': mae, 'smape': smape}

def encode_instruction(args):
    v = []
    v += [args.freq_min/60.0, args.horizon/48.0, min(args.token_budget, 2048)/2048.0]
    v += [float(args.volatility_bin)/5.0]
    v += [1.0 if args.need_explain else 0.0, 1.0 if args.need_ci else 0.0]
    return np.array(v, dtype=np.float32)

def choose_arm(scores, epsilon=0.05):
    if np.random.rand() < epsilon:
        return int(np.random.randint(len(scores)))
    return int(np.argmax(scores))

def prepare_val_probe(val_loader, size):
    idxs = list(range(len(val_loader.dataset)))
    np.random.shuffle(idxs)
    idxs = idxs[:min(size, len(idxs))]
    probe_ds = Subset(val_loader.dataset, idxs)
    probe_loader = torch.utils.data.DataLoader(probe_ds, batch_size=val_loader.batch_size, shuffle=False)
    return probe_loader

def forward_batch_build_inputs(batch, tokenizer, templates, tpl_id, args,
                               news_df, policy_name, policy_kw, sd_kw):
    L, H = args.history_len, args.horizon
    hist_budget = int(args.token_budget * args.token_budget_history_frac)
    news_budget = int(args.token_budget * args.token_budget_news_frac)

    tpl_text = templates[tpl_id]['text']
    prompts, targets = [], []
    for i in range(len(batch['history'])):
        history = batch['history'][i].tolist()
        target = batch['target'][i].tolist()
        t_target = batch['target_time'][i]

        cand = get_candidates(news_df, args.news_time_col, t_target, args.news_window_days, args.news_topM)
        selected = select_news(cand, policy_name, args.news_text_col, policy_kw, sd_kw, args.news_topK)

        hist_str = format_history(history, args.unit, hist_budget, tokenizer)
        news_str = format_news(selected, args.news_text_col, news_budget, tokenizer,
                               summary_method=args.news_summary_method, max_sentences=args.news_max_sentences)

        prompt = build_prompt(tpl_text, L, H, args.unit, hist_str, news_str)
        prompt = prompt + f"\n{SPECIAL_PRED_TOKEN}\n"
        prompts.append(prompt)
        targets.append(target)

    enc = tokenizer(prompts, padding=True, truncation=True, max_length=args.max_seq_len, return_tensors='pt')
    input_ids = enc['input_ids']
    attn = enc['attention_mask']
    labels = torch.tensor(targets, dtype=torch.float32)
    return input_ids, attn, labels

def evaluate_probe(model, tokenizer, probe_loader, templates, tpl_id, args,
                   news_df, policy_name, policy_kw, sd_kw, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in probe_loader:
            input_ids, attn, labels = forward_batch_build_inputs(
                batch, tokenizer, templates, tpl_id, args, news_df, policy_name, policy_kw, sd_kw
            )
            input_ids = input_ids.to(device); attn = attn.to(device); labels = labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=None)
            y_hat = out['pred']
            preds.append(y_hat.detach().cpu().numpy())
            trues.append(labels.cpu().numpy())
    preds = np.concatenate(preds, axis=0); trues = np.concatenate(trues, axis=0)
    m = METRIC_FN[args.reward_metric](trues, preds)
    return m

def main(args):
    set_seed(args.seed)
    device = device_from_id(args.gpu)

    def _read(path):
        if path.endswith('.parquet'): return pd.read_parquet(path)
        return pd.read_csv(path)
    train_df = _read(args.train_file); val_df = _read(args.val_file)
    train_df[args.time_col] = pd.to_datetime(train_df[args.time_col])
    val_df[args.time_col] = pd.to_datetime(val_df[args.time_col])

    train_loader = make_loader(train_df, args.time_col, args.value_col,
                               args.history_len, args.horizon, args.stride, args.batch_size, shuffle=True, id_col=args.id_col)
    val_loader = make_loader(val_df, args.time_col, args.value_col,
                             args.history_len, args.horizon, args.stride, args.batch_size, shuffle=False, id_col=args.id_col)
    probe_loader = prepare_val_probe(val_loader, args.rl_val_probe_size)

    news_df = pd.DataFrame(columns=[args.news_time_col, args.news_text_col])
    if args.news_path:
        news_df = load_news(args.news_path, args.news_time_col, args.news_tz)
    policy_kw = _load_keywords(args.policy_keywords_policy)
    sd_kw = _load_keywords(args.policy_keywords_supplydemand)

    templates = load_templates(args.template_pool)
    allowed_tpl_ids = list(templates.keys()) if args.template_ids is None else args.template_ids

    tokenizer, model = load_llama_lora(
        base_model=args.base_model,
        tokenizer_id=args.tokenizer,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.target_modules,
        load_in_4bit=args.load_in_4bit, gradient_checkpointing=args.gradient_checkpointing,
        max_seq_len=args.max_seq_len, device=device, horizon=args.horizon
    )
    model.to(device)
    model.train()

    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=args.lr, weight_decay=args.weight_decay)

    instr_feat = encode_instruction(args)
    def tpl_features(tid): return np.array([templates[tid].get('has_explain',0)], dtype=np.float32)
    d_tpl = len(instr_feat) + len(tpl_features(allowed_tpl_ids[0]))
    bandit_tpl = LinTS(d_tpl, v=args.ts_v) if args.rl_algo=='lints' else LinUCB(d_tpl, alpha=args.ucb_alpha)

    policy_space = ['recent_topk', 'policy_only', 'supply_demand', 'mixed_alpha']
    if args.news_policy not in policy_space: policy_space.append(args.news_policy)
    def pol_features(i): return np.array([i/(max(1,len(policy_space)-1))], dtype=np.float32)
    d_pol = len(instr_feat) + len(pol_features(0))
    bandit_pol = LinTS(d_pol, v=args.ts_v) if args.rl_algo=='lints' else LinUCB(d_pol, alpha=args.ucb_alpha)

    normalizer = RewardNormalizer(ema=args.reward_ema, use_group_norm=args.domain_reward_norm)
    prev_metric = None
    global_step = 0
    best_metric = float('inf')
    stale_rounds = 0

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            scores_tpl = []
            for tid in allowed_tpl_ids:
                x = np.concatenate([instr_feat, tpl_features(tid)], axis=0)
                s = bandit_tpl.sample_score(x) if isinstance(bandit_tpl, LinTS) else bandit_tpl.ucb_score(x)
                scores_tpl.append(s)
            tpl_idx = choose_arm(scores_tpl, epsilon=args.epsilon)
            tpl_id = allowed_tpl_ids[tpl_idx]

            scores_pol = []
            for i, pol in enumerate(policy_space):
                x = np.concatenate([instr_feat, pol_features(i)], axis=0)
                s = bandit_pol.sample_score(x) if isinstance(bandit_pol, LinTS) else bandit_pol.ucb_score(x)
                scores_pol.append(s)
            pol_idx = choose_arm(scores_pol, epsilon=args.epsilon)
            policy_name = policy_space[pol_idx]

            input_ids, attn, labels = forward_batch_build_inputs(
                batch, tokenizer, templates, tpl_id, args, news_df, policy_name, policy_kw, sd_kw
            )
            input_ids = input_ids.to(device); attn = attn.to(device); labels = labels.to(device)

            steps_this_cycle = max(0, args.rl_cycle_steps)
            if steps_this_cycle > 0:
                for _ in range(steps_this_cycle):
                    model.train()
                    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                    loss = out['loss']
                    loss.backward()
                    if (global_step + 1) % args.grad_accum == 0:
                        optim.step(); optim.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % args.eval_interval == 0:
                        model.eval()
                        metric_now = evaluate_probe(model, tokenizer, probe_loader, templates, tpl_id, args,
                                                    news_df, policy_name, policy_kw, sd_kw, device)
                        if args.reward_mode == 'delta' and prev_metric is not None:
                            r = (prev_metric - metric_now)
                        else:
                            r = -metric_now
                        tok_len = input_ids.size(1)
                        r -= args.reward_len_penalty * float(tok_len)
                        r -= args.reward_k_penalty * float(args.news_topK)
                        r_hat = normalizer.update_and_normalize(r, group_key=(args.region, args.horizon) if args.domain_reward_norm else None)

                        x_tpl = np.concatenate([instr_feat, tpl_features(tpl_id)], axis=0)
                        x_pol = np.concatenate([instr_feat, pol_features(pol_idx)], axis=0)
                        bandit_tpl.update(x_tpl, r_hat)
                        bandit_pol.update(x_pol, r_hat)
                        prev_metric = metric_now
                        pbar.set_postfix({f'val_{args.reward_metric}': f"{metric_now:.4f}"})

                        if metric_now < best_metric - 1e-6:
                            best_metric = metric_now
                            stale_rounds = 0
                            os.makedirs(args.save_dir, exist_ok=True)
                            torch.save({'model': model.state_dict(), 'step': global_step},
                                       os.path.join(args.save_dir, f'best.pt'))
                        else:
                            stale_rounds += 1
                            if stale_rounds >= args.early_stop_patience:
                                print("Early stopping triggered.")
                                return
                    if args.save_interval > 0 and global_step % args.save_interval == 0:
                        os.makedirs(args.save_dir, exist_ok=True)
                        torch.save({'model': model.state_dict(), 'step': global_step},
                                   os.path.join(args.save_dir, f'step{global_step}.pt'))
            else:
                model.eval()
                metric_now = evaluate_probe(model, tokenizer, probe_loader, templates, tpl_id, args,
                                            news_df, policy_name, policy_kw, sd_kw, device)
                r = (prev_metric - metric_now) if (args.reward_mode=='delta' and prev_metric is not None) else -metric_now
                r_hat = normalizer.update_and_normalize(r, group_key=(args.region, args.horizon) if args.domain_reward_norm else None)
                bandit_tpl.update(np.concatenate([instr_feat, tpl_features(tpl_id)], axis=0), r_hat)
                bandit_pol.update(np.concatenate([instr_feat, pol_features(pol_idx)], axis=0), r_hat)
                prev_metric = metric_now
                pbar.set_postfix({f'val_{args.reward_metric}': f"{metric_now:.4f}"})
