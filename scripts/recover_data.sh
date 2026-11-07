#!/bin/bash
source "$(dirname "$0")/common.sh"
GPUS="0,1"
MODEL="bert-base-cased"
DATASET="sst2"
WEIGHT_DIR="results/train_results"
RECOVER_DATA_DIR="data/recover_data"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --weight_dir) WEIGHT_DIR="$2"; shift ;;
        --recover_data_dir) RECOVER_DATA_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
python code/recover_data.py \
    --gpus "$GPUS" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --weight_dir "$WEIGHT_DIR" \
    --recover_data_dir "$RECOVER_DATA_DIR"
