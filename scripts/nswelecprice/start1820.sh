

python run.py \
    --taskName 'NSW2018To2020ElectricityPriceWithNews' \
    --time_col 'SETTLEMENTDATE'\
    --value_col 'RRP'\
    --unit "$/MWh" \
    --description "This dataset records the electricity price data in Australia NSW from 2018 to 2020, collected from National electricity market." \
    --region "Australia, NSW"\
    --news_path dataset/Summarized_news_2015_2020.json \
    --dayFirst True \
    --train_file dataset/2018-2020NSWelecprice/2018To2020NSWData_trainset.csv \
    --val_file dataset/2018-2020NSWelecprice/2018To2020NSWData_valset.csv \
    --test_file dataset/2018-2020NSWelecprice/2018To2020NSWData_testset.csv \
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
