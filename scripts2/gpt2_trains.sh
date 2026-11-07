#!/bin/bash

source "$(dirname "$0")/../scripts/common.sh"

GPUS="0,1"
MODEL="gpt2"
DATASET1="mnli"
DATASET2="qqp"
DATASET3="qnli"
DATASET4="sst2"
output_dir="results/train_results"

./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET1 --output_dir $output_dir 
./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET2 --output_dir $output_dir 
./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET3 --output_dir $output_dir 
./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET4 --output_dir $output_dir 



echo "所有脚本执行完毕！"