# run.py
import argparse
from src.trainer import main as main_train
from src.chatgpt_4o_mini_keyword_generate.gpt_client import run_from_config, stream_from_config
from pathlib import Path
from openai import OpenAI
from textblob import TextBlob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-domain LLaMA Forecasting with RL')

    # ===== Basic =====
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'],
                        help='training precision/mixed precision mode')

    # ===== Data & Instruction =====
    parser.add_argument('--data_root', type=str, default='./dataset', help='root folder for datasets')
    parser.add_argument('--dayFirst', type=bool, default=False, help='Are your datasets in "day-first" format?')
    parser.add_argument('--train_file', type=str, default='', help='train set path (CSV/Parquet)')
    parser.add_argument('--val_file', type=str, default='', help='validation set path')
    parser.add_argument('--test_file', type=str, default='', help='test set path')
    parser.add_argument('--time_col', type=str, default='timestamp', help='time column name')
    parser.add_argument('--value_col', type=str, default='value', help='target/series column name')
    parser.add_argument('--id_col', type=str, default=None, help='optional series ID column for multi-series')
    parser.add_argument('--scaler', type=str, default='standard', choices=['none', 'standard', 'minmax'],
                        help='scaling method for target')

    parser.add_argument('--instruction_json', type=str, default='',
                        help='path to JSON with instruction fields per run/domain')
    parser.add_argument('--freq_min', type=int, default=30, help='sampling frequency in minutes (e.g., 30 for half-hour)')
    parser.add_argument('--region', type=str, default='', help='region/state/country code')
    parser.add_argument('--unit', type=str, default='', help='unit string')
    parser.add_argument('--season', type=str, default='', help='DJF/MAM/JJA/SON or empty')
    parser.add_argument('--volatility_bin_tiers', type=int, default=100, help='the tiers to bin volatility')
    parser.add_argument('--token_budget', type=int, default=1200, help='max tokens for composed prompt')
    parser.add_argument('--val_ema_alpha', type=float, default=0.9, help='EMA alpha for validation loss smoothing')

    # Whether to include explanations in the prompt template
    parser.add_argument('--need_explain', action='store_true', help='include explanations in template')
    parser.add_argument('--need_ci', action='store_true', help='include confidence interval in output')

    # ===== Task windowing =====
    parser.add_argument('--history_len', type=int, default=48, help='steps for history window L')
    parser.add_argument('--horizon', type=int, default=48, help='steps to predict H')
    parser.add_argument('--stride', type=int, default=48, help='sliding stride for training')

    # ===== News retrieval (rule-based) =====
    parser.add_argument('--news_path', type=str, default='', help='path to news store (should be a JSON file)')
    parser.add_argument('--news_time_col', type=str, default='date', help='news timestamp column name')
    parser.add_argument('--news_text_col', type=str, default='content', help='news text/summary column name')
    parser.add_argument('--news_source_col', type=str, default='source', help='news source column (optional)')
    parser.add_argument('--news_tz', type=str, default='', help='timezone for news timestamps')
    parser.add_argument('--news_window_days', type=int, default=1, help='look-back window (days) before target time')
    parser.add_argument('--news_topM', type=int, default=20, help='candidate news cap per sample')
    parser.add_argument('--news_topK', type=int, default=5, help='news K after policy/RL')
    # parser.add_argument('--news_policy', type=str, default='',
    #                     help='rule-based extraction strategy or combinational bandit')
    
    # Keyword files for policy-based news selection
    parser.add_argument('--keyword_path', type=str, default='keywords/kws.txt',
                        help='keyword list for filtering news (one per line)')
    
    # ===== News summarization =====
    # WE ALREADY USED CHATGPT4O TO SUMMARIZE NEWS, SO THIS IS NOT USED
    parser.add_argument('--news_summary_method', type=str, default='none', choices=['none', 'lead3', 'rule'],
                        help='shorten news before inserting to prompt')
    parser.add_argument('--news_max_sentences', type=int, default=3, help='max sentences per selected news')

    # ===== Prompt templates =====
    parser.add_argument('--template_pool', type=str, default='configs/templates.yaml',
                        help='YAML/JSON templates with placeholders')
    # Use template_ids to restrict to a subset of templates
    parser.add_argument('--template_ids', type=str, default='', help='comma-separated template ids to allow (empty=all)')

    #===== Token budget fractions =====
    parser.add_argument('--token_budget_history_frac', type=float, default=0.5, help='budget frac for history')
    parser.add_argument('--token_budget_news_frac', type=float, default=0.4, help='budget frac for news')
    parser.add_argument('--token_budget_instr_frac', type=float, default=0.1, help='budget frac for instruction')

    # ===== LLaMA =====
    parser.add_argument('--base_model', type=str, default='meta-llama/Meta-Llama-3-8B', help='HF model id or local path')
    parser.add_argument('--tokenizer', type=str, default='', help='HF tokenizer id (default: same as base_model)')
    parser.add_argument('--load_in_4bit', action='store_true', help='use 4-bit quantization (QLoRA)')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='enable gradient checkpointing')
    parser.add_argument('--max_seq_len', type=int, default=10000, help='max sequence length')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj',
                        help='comma-separated target module names for LoRA')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for LoRA params')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='warmup ratio')
    parser.add_argument('--batch_size', type=int, default=2, help='micro batch size per device')
    parser.add_argument('--grad_accum', type=int, default=16, help='gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3, help='outer epochs over the dataset')
    parser.add_argument('--max_steps', type=int, default=-1, help='override total steps if >0')
    # parser.add_argument('--eval_interval', type=int, default=200, help='validate every N steps')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='checkpoint dir')
    parser.add_argument('--save_interval', type=int, default=1000, help='save every N steps')

    # ===== RL / Bandit =====
    parser.add_argument('--rl_use', type=int, default=1, help='use RL/bandit for news selection? (0/1)')
    parser.add_argument('--rl_algo', type=str, default='lints', choices=['lints','linucb'], help='bandit algorithm')
    parser.add_argument('--rl_cycle_steps', type=int, default=100, help='short SFT steps per decision cycle (0=no-train)')
    parser.add_argument('--rl_update_times', type=int, default=1, help='update bandits every N cycles')
    parser.add_argument('--rl_val_probe_size', type=int, default=256, help='fixed validation probe size')
    parser.add_argument('--rl_val_probe_frac', type=float, default=0.5, help='fixed validation probe fraction')

    parser.add_argument('--reward_metric', type=str, default='rmse', choices=['mse','mae','loss'], help='reward metric')
    parser.add_argument('--reward_mode', type=str, default='delta', choices=['delta','negative'], help='delta or negative')
    parser.add_argument('--reward_len_penalty', type=float, default=0.0, help='penalty for prompt tokens')
    parser.add_argument('--reward_k_penalty', type=float, default=0.0, help='penalty for K news')
    parser.add_argument('--reward_ema', type=float, default=0.3, help='EMA smoothing of reward')
    parser.add_argument('--domain_reward_norm', action='store_true', help='z-score reward per (domain,horizon) group')
    parser.add_argument('--ucb_alpha', type=float, default=1.0, help='LinUCB alpha')
    parser.add_argument('--ts_v', type=float, default=1.0, help='LinTS prior scale v')
    parser.add_argument('--epsilon', type=float, default=0.05, help='epsilon-greedy fallback')

    # ===== Eval & Logging =====
    parser.add_argument('--early_stop_patience', type=int, default=10, help='patience in eval rounds')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log directory')
    parser.add_argument('--run_name', type=str, default='xl-rl-forecast', help='run name')

    # ===== Keyword Generation =====
    parser.add_argument("--description",type=str,default="",help="描述这个 dataset 的用途，例如 '新州电价数据'")
    parser.add_argument("--keyword_number",type=int,default=10,help="how many keywords to generate")
    parser.add_argument("--select_policy_by",type = str,default = "epoch",choices=["epoch", "batch"], 
                        help = "Select policy/template by epoch-level or batch-level"
    )

    args = parser.parse_args()

    if not args.tokenizer:
        args.tokenizer = args.base_model
    args.target_modules = [s.strip() for s in args.target_modules.split(',') if s.strip()]
    if args.template_ids:
        args.template_ids = [int(x) for x in args.template_ids.split(',') if x.strip()]
    else:
        args.template_ids = None
    s = args.token_budget_history_frac + args.token_budget_news_frac + args.token_budget_instr_frac
    if s > 1.0:
        args.token_budget_history_frac /= s
        args.token_budget_news_frac    /= s
        args.token_budget_instr_frac   /= s

    # # 生成关键词
    
    # description = args.description.strip()
    # text = run_from_config(
    #     config_path="src/chatgpt_4o_mini_keyword_generate/config.json",
    #     kind="generate_keywords",  # 选择 A/B/C
    #     variables={
    #         "description": description if description else "null",
    #         "number": args.keyword_number
    #     },
    #     system="Be concise in your output.",
    #     temperature=0.2,
    # )

    #   # 获取输出文本
    # text = text.strip()

    # # 确保目录存在
    # out_path = Path("keywords/kws.txt")
    # out_path.parent.mkdir(parents=True, exist_ok=True)

    # # 写入文件
    # with open(out_path, "w", encoding="utf-8") as f:
    #     f.write(text)

    # print(f"[Keywords] Have been recorded in {out_path}")


    main_train(args)
