#!/bin/bash
source "$(dirname "$0")/common.sh"
python - <<'EOF'
import sys
import importlib

ok = True
print("Python       :", sys.version.split()[0])
for name, label in [("torch", "PyTorch"), ("torchvision", "torchvision"), ("transformers", "Transformers"),
                    ("datasets", "Datasets"), ("peft", "PEFT"), ("timm", "timm"), ("sklearn", "scikit-learn")]:
    try:
        mod = importlib.import_module(name)
        print(f"{label:<13}:", mod.__version__)
    except ImportError:
        print(f"{label:<13}: 未安装")
        if name in ("torch", "transformers", "sklearn"):
            ok = False
try:
    import torch
    print("CUDA 可用    :", torch.cuda.is_available())
    print("GPU 数量     :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory / 1024 ** 3:.0f} GB")
    ok = ok and torch.cuda.is_available()
except Exception as e:
    print("GPU 检查失败 :", e)
    ok = False
print("Successfully finished basic test" if ok else "Basic test failed，请检查运行环境")
sys.exit(0 if ok else 1)
EOF
