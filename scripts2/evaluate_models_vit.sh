#!/bin/bash
source "$(dirname "$0")/../scripts/common.sh"
for DATASET in cifar_10 cifar_100 food101; do
    ./scripts/evaluate_vit.sh --dataset "$DATASET"
done
echo "所有脚本执行完毕！"
