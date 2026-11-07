# 泛在互联操作系统的大语言模型参数高效混淆系统 V1.0

本目录是按《设计文档》组织的代码版本，包含 ArrowCloak 参数混淆模块和基于 ArrowMatch 的内部安全评估模块，支持 BERT、GPT-2、GPT-2 XL、ViT 四类模型，运行在 64 位 Linux + NVIDIA GPU 环境下。

## 目录结构

```
install1.sh / install2.sh   创建运行环境（文本模型 / ViT），详细输出写入 logs/
scripts/common.sh           各脚本共用的环境设置（镜像地址、屏蔽第三方库警告、检查点命名）
scripts/check_env.sh        环境自检
scripts/train*.sh           私有模型训练
scripts/recover_data.sh     恢复数据集构建
scripts/arrowcloak*.sh      ArrowCloak 混淆与 ArrowMatch 安全评估
scripts/evaluate*.sh        综合评估
scripts2/                   批量运行脚本
train/  code/  utils/       Python 程序与算法库
```

## 运行流程（以 BERT + SST-2 为例，与设计文档第 3.4 节、第 5 章一一对应）

```
# 1. 环境部署与自检
export HF_ENDPOINT="https://hf-mirror.com"
./install1.sh
conda activate arrowcloak-env1
./scripts/check_env.sh

# 2. 私有模型训练（结束后自动把最后一个检查点命名为 final_checkpoint）
./scripts/train.sh --gpus 0,1 --model bert-base-cased --dataset sst2

# 3. 恢复数据集构建
./scripts/recover_data.sh --gpus 0,1 --model bert-base-cased --dataset sst2

# 4. ArrowCloak 混淆与安全评估
./scripts/arrowcloak.sh --gpus 0,1 --dataset sst2

# 5. 综合评估
./scripts/evaluate.sh --gpus 0,1 --dataset sst2
```

其他模型换用对应脚本：GPT-2 为 train.sh（--model gpt2）、arrowcloak_gpt2.sh、evaluate_gpt2.sh；GPT-2 XL 为 train_gpt2_xl.sh、arrowcloak_gpt2_xl.sh、evaluate_gpt2_xl.sh；ViT 使用 arrowcloak-env2 环境，脚本为 train_vit.sh、arrowcloak_vit.sh、evaluate_vit.sh。recover_data.sh 通过 --model 参数（bert-base-cased、gpt2、gpt2-xl、vit）支持全部模型。
