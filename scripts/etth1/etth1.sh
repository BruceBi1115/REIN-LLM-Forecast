#!/bin/bash

############################
# 固定不变的参数
############################
GPU=1
TIME_COL="date"
VALUE_COL="OT"
FREQ_MIN=60
UNIT="$/MWh"
DESC="This dataset records the electricity transformer temperature."
REGION="China"

TRAIN_FILE="dataset/ETTh1/ETTh1_trainset.csv"
VAL_FILE="dataset/ETTh1/ETTh1_valset.csv"
TEST_FILE="dataset/ETTh1/ETTh1_testset.csv"

NEWS_PATH="dataset/Summarized_news_2015_2020.json"
NEWS_TEXT_COL="summary_response"
NEWS_TIME_COL="date"
KEYWORD_PATH="keywords/kws_etth1.txt"

EPOCHS=1
KEYWORD_NUM=20
NEWS_WINDOW=3
NEWS_TOPM=20
NEWS_TOPK=5
BATCH_SIZE=1

RL_ALGO="lints"
REWARD_METRIC="mse"
RL_CYCLE_STEPS=1
SELECT_POLICY="epoch"

BASE_TASK="TestEtth1withElecpriceNewsfrom2015To2020"

############################
# 实验配置
# name | rl_use | use_news
############################
CONFIGS=(
  # "Normal|1|1"
  "NoRL|0|1"
  # "NoNews|1|0"
  "Pure|0|0"
)

############################
# 主循环
############################
for CFG in "${CONFIGS[@]}"; do
  IFS="|" read TAG RL_USE USE_NEWS <<< "$CFG"

  if [ "$TAG" == "Normal" ]; then
    TASK_NAME="${BASE_TASK}"
  else
    TASK_NAME="[${TAG}]${BASE_TASK}"
  fi

  CMD="python run.py \
    --taskName \"${TASK_NAME}\" \
    --freq_min ${FREQ_MIN} \
    --time_col ${TIME_COL} \
    --value_col ${VALUE_COL} \
    --gpu ${GPU} \
    --unit \"${UNIT}\" \
    --description \"${DESC}\" \
    --region ${REGION} \
    --train_file ${TRAIN_FILE} \
    --val_file ${VAL_FILE} \
    --test_file ${TEST_FILE} \
    --epochs ${EPOCHS} \
    --keyword_number ${KEYWORD_NUM} \
    --news_window_days ${NEWS_WINDOW} \
    --news_topM ${NEWS_TOPM} \
    --news_topK ${NEWS_TOPK} \
    --batch_size ${BATCH_SIZE} \
    --rl_use ${RL_USE} \
    --rl_algo ${RL_ALGO} \
    --reward_metric ${REWARD_METRIC} \
    --rl_cycle_steps ${RL_CYCLE_STEPS} \
    --select_policy_by ${SELECT_POLICY}
  "

  if [ "$USE_NEWS" -eq 1 ]; then
    CMD+=" \
      --news_path ${NEWS_PATH} \
      --news_text_col ${NEWS_TEXT_COL} \
      --news_time_col ${NEWS_TIME_COL} \
      --keyword_path ${KEYWORD_PATH}
    "
  fi

  pure_template_config="configs/pure_template.yaml"
  norl_template_config="configs/norl_template.yaml"
  if [ "$RL_USE" -eq 0 ]; then
      CMD+=" \
      --template_pool ${norl_template_config}
      "
    if [ "$USE_NEWS" -eq 0 ]; then
      CMD+=" \
      --template_pool ${pure_template_config}
      "
    fi
  fi

  echo "========================================"
  echo "Running: ${TASK_NAME}"
  echo "RL_USE=${RL_USE}, USE_NEWS=${USE_NEWS}"
  echo "========================================"

  eval $CMD
done
