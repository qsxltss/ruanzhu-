#!/bin/bash

source "$(dirname "$0")/../scripts/common.sh"

DATASET1="cifar_10"
DATASET2="cifar_100"
DATASET3="food101"
OUTPUT_DIR="results/train_results/ViT"

./scripts/train_vit.sh --dataset $DATASET1 --output_dir $OUTPUT_DIR
./scripts/train_vit.sh --dataset $DATASET2 --output_dir $OUTPUT_DIR
./scripts/train_vit.sh --dataset $DATASET3 --output_dir $OUTPUT_DIR



echo "所有脚本执行完毕！"