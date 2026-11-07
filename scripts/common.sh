#!/bin/bash
# 各运行脚本共用的环境设置：使用镜像站，屏蔽第三方库的警告信息和下载进度条
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY="error"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM="false"
export TF_CPP_MIN_LOG_LEVEL=3

# 把目录中最后一个检查点命名为 final_checkpoint，供后续环节读取
finalize_checkpoint() {
    local dir="$1"
    local last
    last=$(ls -d "$dir"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$last" ]; then
        return
    fi
    # 本次运行没有产生比 final_checkpoint 更新的检查点时（如直接复用已有结果），保持不变
    if [ -d "$dir/final_checkpoint" ] && [ ! "$last" -nt "$dir/final_checkpoint" ]; then
        return
    fi
    if [ -d "$dir/final_checkpoint" ]; then
        mv "$dir/final_checkpoint" "$dir/final_checkpoint.bak_$(date +%Y%m%d_%H%M%S)"
    fi
    mv "$last" "$dir/final_checkpoint"
    echo "最终模型已保存：$dir/final_checkpoint"
}

# 模型名到结果子目录名的对应关系
model_dir_name() {
    case "$1" in
        bert-base-cased) echo "bert" ;;
        gpt2) echo "gpt2_base" ;;
        gpt2-xl) echo "gpt2_xl" ;;
        vit) echo "ViT" ;;
        *) echo "Unknown model: $1" >&2; exit 1 ;;
    esac
}
