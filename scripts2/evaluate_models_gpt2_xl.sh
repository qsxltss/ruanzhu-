#!/bin/bash
source "$(dirname "$0")/../scripts/common.sh"
GPUS="0,1"
for DATASET in sst2; do
    ./scripts/evaluate_gpt2_xl.sh --gpus "$GPUS" --dataset "$DATASET"
done
echo "所有脚本执行完毕！"
