#!/bin/bash
source "$(dirname "$0")/../scripts/common.sh"
GPUS="0,1"
for DATASET in mnli qqp qnli sst2; do
    ./scripts/evaluate_gpt2.sh --gpus "$GPUS" --dataset "$DATASET"
done
echo "所有脚本执行完毕！"
