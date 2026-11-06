# 泛在互联操作系统的大语言模型参数高效混淆系统 V1.0

本目录是按《设计文档》组织的代码版本，包含 ArrowCloak 参数混淆模块和基于 ArrowMatch 的内部安全评估模块，支持 BERT、GPT-2、GPT-2 XL、ViT 四类模型。运行在 64 位 Linux + NVIDIA GPU 环境下。

## 目录结构

```
install1.sh / install2.sh   创建运行环境（文本模型 / ViT）
train/                      私有模型训练程序
code/arrowcloak*.py         ArrowCloak 混淆与 ArrowMatch 安全评估程序
code/evaluate_model*.py     综合评估程序
utils/methods*.py           ob_arrowcloak（混淆）、arrowmatch_recover（恢复评估）等算法接口
utils/utils*.py             数据准备、恢复数据集构建、置换匹配、模型加载等工具接口
scripts/                    单任务运行脚本
scripts2/                   环境自检与批量运行脚本
```

## 运行环境

```
export HF_ENDPOINT="https://hf-mirror.com"
./install1.sh && conda activate arrowcloak-env1      # BERT / GPT-2 / GPT-2 XL
./install2.sh && conda activate arrowcloak-env2      # ViT
```

## 基本流程（以 BERT + SST-2 为例）

```
# 1. 训练私有模型，完成后把最后一个 checkpoint-编号 目录改名为 final_checkpoint
./scripts/train.sh --gpus 0,1 --model bert-base-cased --dataset sst2 --output_dir results/train_results

# 2. ArrowCloak 混淆 + ArrowMatch 安全评估，完成后同样把最后一个检查点改名为 final_checkpoint
./scripts/arrowcloak.sh --gpus 0,1 --dataset sst2

# 3. 综合评估（白盒、黑盒、混淆模型、ArrowMatch 恢复模型）
python code/evaluate_model.py --dataset sst2 --gpus 0,1
```

ViT 版本使用 `scripts/train_vit.sh`、`scripts/arrowcloak_vit.sh`、`code/evaluate_model_vit.py`，模型保存为 ckpt.t7，不需要改名。批量运行见 `scripts2/` 目录。
