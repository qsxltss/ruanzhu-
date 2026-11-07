#!/bin/bash

source "$(dirname "$0")/../scripts/common.sh"

GPUS="0,1"
DATASET4="sst2"
output_dir="results/train_results"

./scripts/train_gpt2_xl.sh --gpus $GPUS --dataset $DATASET4 --output_dir $output_dir 


echo "所有脚本执行完毕！"