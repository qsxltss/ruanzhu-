#!/bin/bash
source "$(dirname "$0")/common.sh"
GPUS="0,1"
DATASET="sst2"
OUTPUT_DIR="evaluate_results"
EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --result_of_*) EXTRA_ARGS+=("$1" "$2"); shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
python code/evaluate_model.py \
    --gpus "$GPUS" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}" || exit 1
RESULT_DIR="$OUTPUT_DIR/bert/$DATASET"
echo "========================= 评估结果汇总 ========================="
for name in whitebox blackbox obfus arrowcloak; do
    f="$RESULT_DIR/${name}_results.txt"
    if [ -f "$f" ]; then
        tail -n 1 "$f"
        echo
    fi
done
echo "结果文件目录：$RESULT_DIR"
