    # --train_file dataset/2024NSWelecprice/2024NSWelecprice_trainset.csv \
    # --val_file dataset/2024NSWelecprice/2024NSWelecprice_valset.csv \
    # --test_file dataset/2024NSWelecprice/2024NSWelecprice_testset.csv \

python run.py \
    --taskName 'NSW2024To2025ElectricityPriceWithNews' \
    --time_col 'SETTLEMENTDATE'\
    --value_col 'RRP'\
    --unit "$/MWh" \
    --description "This dataset records the electricity price data in Australia NSW from 2024 to 2025, collected from National electricity market." \
    --region "Australia, NSW"\
    --news_path dataset/Summarized_news_2024_2025.json \
    --train_file dataset/2024NSWelecprice/2024NSWelecprice_trainset.csv \
    --val_file dataset/2024NSWelecprice/2024NSWelecprice_valset.csv \
    --test_file dataset/2024NSWelecprice/2024NSWelecprice_testset.csv \
    --news_text_col 'summary_response' \
    --news_time_col 'date' \
    --keyword_path 'keywords/kws.txt' \
    --epochs 30\
    --keyword_number 20 \
    --news_window_days 7 \
    --news_topM 20 \
    --news_topK 5 \
    --batch_size 1\
    --rl_use 1 \
    --rl_algo "lints" \
    --reward_metric "mse" \
    --rl_cycle_steps 1 \
    --select_policy_by "epoch"
