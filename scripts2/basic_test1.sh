#!/bin/bash
source "$(dirname "$0")/../scripts/common.sh"
GPUS="0,1"
DATASET="sst2"
./scripts/check_env.sh || exit 1
./scripts/train.sh --gpus $GPUS --model bert-base-cased --dataset $DATASET || exit 1
./scripts/recover_data.sh --gpus $GPUS --model bert-base-cased --dataset $DATASET || exit 1
./scripts/arrowcloak.sh --gpus $GPUS --dataset $DATASET || exit 1
./scripts/evaluate.sh --gpus $GPUS --dataset $DATASET || exit 1
echo "Successfully finished basic test1"
