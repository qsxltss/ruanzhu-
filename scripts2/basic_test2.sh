#!/bin/bash
source "$(dirname "$0")/../scripts/common.sh"
DATASET="cifar_100"
./scripts/check_env.sh || exit 1
./scripts/train_vit.sh --dataset $DATASET || exit 1
./scripts/recover_data.sh --model vit --dataset $DATASET || exit 1
./scripts/arrowcloak_vit.sh --dataset $DATASET || exit 1
./scripts/evaluate_vit.sh --dataset $DATASET || exit 1
echo "Successfully finished basic test2"
