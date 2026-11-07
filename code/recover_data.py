import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser(description="build recover dataset")
parser.add_argument("--model", default="bert-base-cased", type=str, help="bert-base-cased, gpt2, gpt2-xl or vit")
parser.add_argument("--dataset", default="sst2", type=str, help="dataset")
parser.add_argument("--max_length", default=512, type=int, help="Max sequence length with padding")
parser.add_argument("--bs", default=None, type=int, help="batch size")
parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids")
parser.add_argument("--weight_dir", default="results/train_results", type=str, help="weight directory")
parser.add_argument("--recover_data_dir", default="data/recover_data", type=str, help="data for recovering finetune")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.model == "vit" else args.gpus
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_names = {"bert-base-cased": "bert", "gpt2": "gpt2_base", "gpt2-xl": "gpt2_xl", "vit": "ViT"}
if args.model not in model_names:
    raise ValueError("Invalid model name")
model_name = model_names[args.model]
args.recover_data_dir = f"{args.recover_data_dir}/{model_name}/{args.dataset}"
os.makedirs(args.recover_data_dir, exist_ok=True)


def build_text():
    import torch
    from datasets import load_dataset
    task_to_keys = {
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "sst2": ("sentence", None),
    }
    if args.dataset not in task_to_keys:
        raise ValueError("Invalid dataset")
    if args.model == "bert-base-cased":
        from utils.utils import set_seed, prepare_data, prepare_recover_data
    elif args.model == "gpt2":
        from utils.utils_gpt2 import set_seed, prepare_data, prepare_recover_data
    else:
        from utils.utils_gpt2_xl import set_seed, prepare_data, prepare_recover_data
    bs = args.bs or (4 if args.model == "gpt2-xl" else 32)
    path = f"{args.recover_data_dir}/recover_data.json"
    if os.path.exists(path):
        recover_dataset = load_dataset("json", data_files=path)["train"]
        print(f"恢复数据集已存在：{path}，共 {len(recover_dataset)} 条，直接加载")
        return
    set_seed()
    num_labels = 3 if args.dataset == "mnli" else 2
    validation_key = "validation_matched" if args.dataset == "mnli" else "validation"
    sentence1_key, sentence2_key = task_to_keys[args.dataset]
    print("Preparing data..")
    trainset, evalset, tokenizer = prepare_data(args.dataset, args.model, validation_key, sentence1_key, sentence2_key, args.max_length)
    print("Loading private model..")
    weight_dir = f"{args.weight_dir}/{model_name}/{args.dataset}/final_checkpoint"
    if args.model == "bert-base-cased":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(weight_dir, num_labels=num_labels, use_safetensors=True)
    elif args.model == "gpt2":
        from transformers import GPT2ForSequenceClassification
        model = GPT2ForSequenceClassification.from_pretrained(weight_dir, num_labels=num_labels, use_safetensors=True)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        from peft import LoraConfig
        from safetensors.torch import load_file
        from utils.utils_gpt2_xl import adjust_lora_model
        weight1 = load_file(weight_dir + "/model-00001-of-00002.safetensors")
        weight2 = load_file(weight_dir + "/model-00002-of-00002.safetensors")
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["c_fc", "c_attn", "c_proj"])
        model = adjust_lora_model('gpt2-xl', lora_config=lora_config, num_labels=num_labels, weight1=weight1, weight2=weight2)
        model.config.pad_token_id = tokenizer.pad_token_id
    print("Building recover data..")
    prepare_recover_data(model, trainset, bs, path, ratio=0.01)
    recover_dataset = load_dataset("json", data_files=path)["train"]
    print(f"恢复数据集已保存：{path}，共 {len(recover_dataset)} 条")


def build_vit():
    import torch
    import timm
    from utils.utils_vit import set_seed, prepare_data, prepare_recover_data
    num_classes_map = {"cifar_10": 10, "cifar_100": 100, "food101": 101}
    if args.dataset not in num_classes_map:
        raise ValueError("Invalid dataset name")
    path = f"{args.recover_data_dir}/recover_dataset.pth"
    if os.path.exists(path):
        recover_dataset = torch.load(path)
        print(f"恢复数据集已存在：{path}，共 {len(recover_dataset)} 条，直接加载")
        return
    set_seed()
    args.bs = args.bs or 128
    print("==> Preparing data..")
    trainloader, testloader, num_classes = prepare_data("./data/datasets", args, 224)
    print("==> Loading private model..")
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes_map[args.dataset])
    checkpoint = torch.load(f"{args.weight_dir}/{model_name}/{args.dataset}/ckpt.t7")
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("==> Building recover data..")
    prepare_recover_data(model, trainloader, num_classes, device, args.recover_data_dir, ratio=0.01)
    recover_dataset = torch.load(path)
    print(f"恢复数据集已保存：{path}，共 {len(recover_dataset)} 条")


if args.model == "vit":
    build_vit()
else:
    build_text()
