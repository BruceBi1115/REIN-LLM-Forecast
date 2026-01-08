# # sh scripts/nswelecprice/elecpricenorl.sh
# # sh scripts/nswelecprice/elecpricepure.sh

# python run.py \
#     --taskName '[NoNews]NSW2015To2016ElectricityPriceWithNews' \
#     --time_col 'SETTLEMENTDATE'\
#     --value_col 'RRP'\
#     --unit "$/MWh" \
#     --description "This dataset records the electricity price data in Australia NSW from 2015 to 2016, collected from National electricity market." \
#     --region "Australia, NSW"\
#     --dayFirst True \
#     --train_file dataset/2015-2016NSWelecprice/2015To2016NSWData_trainset.csv \
#     --val_file dataset/2015-2016NSWelecprice/2015To2016NSWData_valset.csv \
#     --test_file dataset/2015-2016NSWelecprice/2015To2016NSWData_testset.csv \
#     --news_text_col 'summary_response' \
#     --news_time_col 'date' \
#     --keyword_path 'keywords/kws.txt' \
#     --epochs 1\
#     --keyword_number 20 \
#     --news_window_days 7 \
#     --news_topM 20 \
#     --news_topK 5 \
#     --batch_size 1\
#     --rl_use 1 \
#     --rl_algo "lints" \
#     --reward_metric "mse" \
#     --rl_cycle_steps 1 \
#     --select_policy_by "epoch"

python run.py \
    --taskName 'NSW2015To2016ElectricityPriceWithNews' \
    --time_col 'SETTLEMENTDATE'\
    --value_col 'RRP'\
    --unit "$/MWh" \
    --description "This dataset records the electricity price data in Australia NSW from 2015 to 2016, collected from National electricity market." \
    --region "Australia, NSW"\
    --news_path dataset/Summarized_news_2015_2020.json \
    --dayFirst True \
    --train_file dataset/2015-2016NSWelecprice/2015To2016NSWData_trainset.csv \
    --val_file dataset/2015-2016NSWelecprice/2015To2016NSWData_valset.csv \
    --test_file dataset/2015-2016NSWelecprice/2015To2016NSWData_testset.csv \
    --news_text_col 'summary_response' \
    --news_time_col 'date' \
    --keyword_path 'keywords/kws.txt' \
    --epochs 1\
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
