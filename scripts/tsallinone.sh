############################################
# Global fixed args (shared across datasets)
############################################
GPU=1
TIME_COL="date"
VALUE_COL="OT"
UNIT="$/MWh"
REGION="China"

NEWS_PATH="dataset/Summarized_news_2015_2020.json"
NEWS_TEXT_COL="summary_response"
NEWS_TIME_COL="date"

EPOCHS=30
KEYWORD_NUM=20
NEWS_WINDOW=7
NEWS_TOPM=20
NEWS_TOPK=5
BATCH_SIZE=1

RL_ALGO="lints"
REWARD_METRIC="mse"
RL_CYCLE_STEPS=1
SELECT_POLICY="epoch"

# Optional: control output / logging
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

############################################
# Ablations
# tag | rl_use | use_news
############################################
CONFIGS=(
  "Normal|1|1"
  "NoRL|0|1"
  "NoNews|1|0"
  "Pure|0|0"
)

############################################
# Datasets
# key | base_task | train | val | test | keyword_path | description
############################################
DATASETS=(
  "60|electricity|TestElectricityDemandwithElecpriceNewsfrom2015To2020|dataset/electricity/electricity_trainset.csv|dataset/electricity/electricity_valset.csv|dataset/electricity/electricity_testset.csv|keywords/kws_electricity.txt|This dataset records the electricity demand."
  "60|ETTh1|TestEtth1withElecpriceNewsfrom2015To2020|dataset/ETTh1/ETTh1_trainset.csv|dataset/ETTh1/ETTh1_valset.csv|dataset/ETTh1/ETTh1_testset.csv|keywords/kws_etth1.txt|This dataset records the electricity transformer temperature."
  "60|ETTh2|TestEtth2withElecpriceNewsfrom2015To2020|dataset/ETTh2/ETTh2_trainset.csv|dataset/ETTh2/ETTh2_valset.csv|dataset/ETTh2/ETTh2_testset.csv|keywords/kws_etth2.txt|This dataset records the electricity transformer temperature."
  "15|ETTm1|TestEttm1withElecpriceNewsfrom2015To2020|dataset/ETTm1/ETTm1_trainset.csv|dataset/ETTm1/ETTm1_valset.csv|dataset/ETTm1/ETTm1_testset.csv|keywords/kws_ettm1.txt|This dataset records the electricity transformer temperature."
  "15|ETTm2|TestEttm2withElecpriceNewsfrom2015To2020|dataset/ETTm2/ETTm2_trainset.csv|dataset/ETTm2/ETTm2_valset.csv|dataset/ETTm2/ETTm2_testset.csv|keywords/kws_ettm2.txt|This dataset records the electricity transformer temperature."
  "10|weather|TestWeatherwithElecpriceNewsfrom2015To2020|dataset/weather/weather_trainset.csv|dataset/weather/weather_valset.csv|dataset/weather/weather_testset.csv|keywords/kws_weather.txt|This dataset records the weather parameters."
)

############################################
# Main loop: dataset x ablation
############################################
for D in "${DATASETS[@]}"; do
  IFS="|" read -r FREQ_MIN DATA_KEY BASE_TASK TRAIN_FILE VAL_FILE TEST_FILE KEYWORD_PATH DESC <<< "$D"

  for CFG in "${CONFIGS[@]}"; do
    IFS="|" read -r TAG RL_USE USE_NEWS <<< "$CFG"

    if [[ "$TAG" == "Normal" ]]; then
      TASK_NAME="${BASE_TASK}"
    else
      TASK_NAME="[${TAG}]${BASE_TASK}"
    fi

    # Per-run log file (safe filename)
    SAFE_TASK="$(echo "${TASK_NAME}" | sed 's/[\/[:space:]]\+/_/g' | sed 's/[^A-Za-z0-9_\-\[\]]/_/g')"
    RUN_LOG="${LOG_DIR}/${DATA_KEY}__${SAFE_TASK}.log"

    CMD=(python run.py
      --taskName "${TASK_NAME}"
      --freq_min "${FREQ_MIN}"
      --time_col "${TIME_COL}"
      --value_col "${VALUE_COL}"
      --gpu "${GPU}"
      --unit "${UNIT}"
      --description "${DESC}"
      --region "${REGION}"
      --train_file "${TRAIN_FILE}"
      --val_file "${VAL_FILE}"
      --test_file "${TEST_FILE}"
      --epochs "${EPOCHS}"
      --keyword_number "${KEYWORD_NUM}"
      --news_window_days "${NEWS_WINDOW}"
      --news_topM "${NEWS_TOPM}"
      --news_topK "${NEWS_TOPK}"
      --batch_size "${BATCH_SIZE}"
      --rl_use "${RL_USE}"
      --rl_algo "${RL_ALGO}"
      --reward_metric "${REWARD_METRIC}"
      --rl_cycle_steps "${RL_CYCLE_STEPS}"
      --select_policy_by "${SELECT_POLICY}"
    )

    if [[ "${USE_NEWS}" -eq 1 ]]; then
      CMD+=(--news_path "${NEWS_PATH}"
           --news_text_col "${NEWS_TEXT_COL}"
           --news_time_col "${NEWS_TIME_COL}"
           --keyword_path "${KEYWORD_PATH}")
    fi

    echo "============================================================"
    echo "Dataset: ${DATA_KEY}"
    echo "Task:    ${TASK_NAME}"
    echo "RL_USE=${RL_USE}, USE_NEWS=${USE_NEWS}"
    echo "Log:     ${RUN_LOG}"
    echo "============================================================"

    # Run + tee log
    "${CMD[@]}" 2>&1 | tee "${RUN_LOG}"
  done
done
