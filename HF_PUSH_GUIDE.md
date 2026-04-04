# Hugging Face Model Push Guide

This guide provides instructions for pushing Assignment 5 models to Hugging Face Hub.

## Quick Start (Automated)

If you have a valid HF_TOKEN set up:

```bash
cd /csehome/m25csa007/m25csa007/assignement5
source .env  # Ensure HF_TOKEN is set
python push_models_to_hf.py
```

## Manual Push Instructions

### Q1: ViT-S LoRA Best Model (Optuna)

**Repository**: https://huggingface.co/duckquack123/vit-s-cifar100-lora-best
**Repository**: https://huggingface.co/DuckyDuck123/vit-s-cifar100-lora-best

Using Python script (from Q1/):
```bash
cd Q1
export HF_TOKEN="your_token_here"
python push_to_hub.py \
  --checkpoint checkpoints/lora_r8_a4_d0.1_best \
  --model_name "duckquack123/vit-s-cifar100-lora-best" \
  --base_model "WinKawaks/vit-small-patch16-224"
```

Using `huggingface_hub` directly:
```bash
cd /csehome/m25csa007/m25csa007/assignement5

python << 'EOF'
from huggingface_hub import login, upload_file
import os

# Login with token
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

# Upload adapter files
files = [
    "Q1/checkpoints/lora_r8_a4_d0.1_best/adapter_config.json",
    "Q1/checkpoints/lora_r8_a4_d0.1_best/adapter_model.safetensors",
    "Q1/checkpoints/lora_r8_a4_d0.1_best/README.md"
]

for file in files:
    upload_file(
        path_or_fileobj=file,
        path_in_repo=os.path.basename(file),
        repo_id="duckquack123/vit-s-cifar100-lora-best",
        token=HF_TOKEN
    )
    print(f"✅ Uploaded: {file}")

print("🎉 Model pushed successfully!")
EOF
```

### Q2: ResNet-18 Clean Classifier

**Repository**: https://huggingface.co/duckquack123/resnet18-cifar10-clean
**Repository**: https://huggingface.co/DuckyDuck123/resnet18-cifar10-clean

```bash
python << 'EOF'
from huggingface_hub import login, upload_file
import os

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

upload_file(
    path_or_fileobj="Q2/checkpoints/resnet18_cifar10_best.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="duckquack123/resnet18-cifar10-clean",
    token=HF_TOKEN
)
print("✅ ResNet-18 model uploaded!")
EOF
```

### Q2: PGD Adversarial Detector

**Repository**: https://huggingface.co/duckquack123/resnet34-pgd-detector
**Repository**: https://huggingface.co/DuckyDuck123/resnet34-pgd-detector

```bash
python << 'EOF'
from huggingface_hub import login, upload_file
import os

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

upload_file(
    path_or_fileobj="Q2/checkpoints/detector_pgd_best.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="duckquack123/resnet34-pgd-detector",
    token=HF_TOKEN
)
print("✅ PGD detector uploaded!")
EOF
```

### Q2: BIM Adversarial Detector

**Repository**: https://huggingface.co/duckquack123/resnet34-bim-detector
**Repository**: https://huggingface.co/DuckyDuck123/resnet34-bim-detector

```bash
python << 'EOF'
from huggingface_hub import login, upload_file
import os

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

upload_file(
    path_or_fileobj="Q2/checkpoints/detector_bim_best.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="duckquack123/resnet34-bim-detector",
    token=HF_TOKEN
)
print("✅ BIM detector uploaded!")
EOF
```

## Setup HF_TOKEN

Get your token from: https://huggingface.co/settings/tokens

Add to `.env`:
```bash
export HF_TOKEN="hf_your_token_here"
```

Then load it:
```bash
source .env
echo "Token set: ${HF_TOKEN:0:10}..."
```

## Verify Uploads

After pushing, verify models are available:

```bash
python << 'EOF'
from huggingface_hub import HfApi

api = HfApi()

models = [
    "duckquack123/vit-s-cifar100-lora-best",
    "duckquack123/resnet18-cifar10-clean",
    "duckquack123/resnet34-pgd-detector",
    "duckquack123/resnet34-bim-detector"
]

for model in models:
    try:
        info = api.repo_info(repo_id=model)
        print(f"✅ {model}: {info.siblings.__len__()} files")
    except Exception as e:
        print(f"❌ {model}: Not found or error - {str(e)[:50]}")
EOF
```

## Model Details

### Q1: vit-s-cifar100-lora-best
- **Type**: LoRA Adapter (PEFT format)
- **Base Model**: WinKawaks/vit-small-patch16-224
- **Dataset**: CIFAR-100
- **Test Accuracy**: 90.69%
- **Validation Accuracy**: 91.05%
- **Rank**: 8
- **Alpha**: 4
- **Dropout**: 0.1
- **Trainable Parameters**: 259,684 (98.8% reduction)
- **Files**: adapter_config.json, adapter_model.safetensors

### Q2: resnet18-cifar10-clean
- **Type**: PyTorch Model
- **Architecture**: ResNet-18
- **Dataset**: CIFAR-10
- **Training**: From scratch (no pretraining)
- **Test Accuracy**: 89.86%
- **Size**: 43 MB
- **File**: pytorch_model.bin (or resnet18_cifar10_best.pt)

### Q2: resnet34-pgd-detector
- **Type**: PyTorch Binary Classifier
- **Architecture**: ResNet-34
- **Task**: Detect PGD adversarial images
- **Binary Classes**: Clean (0), PGD-Adversarial (1)
- **Detection Accuracy**: 98.30%
- **Size**: 82 MB
- **File**: pytorch_model.bin (or detector_pgd_best.pt)

### Q2: resnet34-bim-detector
- **Type**: PyTorch Binary Classifier
- **Architecture**: ResNet-34
- **Task**: Detect BIM adversarial images
- **Binary Classes**: Clean (0), BIM-Adversarial (1)
- **Detection Accuracy**: 86.02%
- **Size**: 82 MB
- **File**: pytorch_model.bin (or detector_bim_best.pt)

## Loading Models from Hugging Face

### Q1: Load LoRA Adapter

```python
from peft import PeftModel
from transformers import ViTForImageClassification

base_model = ViTForImageClassification.from_pretrained(
    "WinKawaks/vit-small-patch16-224"
)

model = PeftModel.from_pretrained(
    base_model,
    "duckquack123/vit-s-cifar100-lora-best"
)

# Optional: merge adapters into base model
model = model.merge_and_unload()

# Use for inference
```

### Q2: Load Classifiers

```python
import torch
from torchvision.models import resnet18, resnet34

# ResNet-18 Classifier
model = resnet18(num_classes=10)
checkpoint = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/duckquack123/resnet18-cifar10-clean/resolve/main/pytorch_model.bin",
    progress=True
)
model.load_state_dict(checkpoint)

# PGD Detector
detector_pgd = resnet34(num_classes=2)
checkpoint_pgd = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/duckquack123/resnet34-pgd-detector/resolve/main/pytorch_model.bin",
    progress=True
)
detector_pgd.load_state_dict(checkpoint_pgd)

# BIM Detector
detector_bim = resnet34(num_classes=2)
checkpoint_bim = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/duckquack123/resnet34-bim-detector/resolve/main/pytorch_model.bin",
    progress=True
)
detector_bim.load_state_dict(checkpoint_bim)
```

## Troubleshooting

### Invalid Token Error
- Verify token from https://huggingface.co/settings/tokens
- Ensure token hasn't expired
- Check token format starts with `hf_`

### Upload Size Limit
- GitHub has a 100MB per-file limit
- HF Hub has higher limits (multiple GB)
- Large files should use Git LFS

### Authentication Issues
```bash
# Clear old credentials
rm ~/.huggingface/token

# Re-login
export HF_TOKEN="your_token"
huggingface-cli login --token $HF_TOKEN
```

## Links

**Q1 Model**: https://huggingface.co/duckquack123/vit-s-cifar100-lora-best

**Q2 Classifier**: https://huggingface.co/duckquack123/resnet18-cifar10-clean

**Q2 PGD Detector**: https://huggingface.co/duckquack123/resnet34-pgd-detector

**Q2 BIM Detector**: https://huggingface.co/duckquack123/resnet34-bim-detector
**Q1 Model**: https://huggingface.co/DuckyDuck123/vit-s-cifar100-lora-best

**Q2 Classifier**: https://huggingface.co/DuckyDuck123/resnet18-cifar10-clean

**Q2 PGD Detector**: https://huggingface.co/DuckyDuck123/resnet34-pgd-detector

**Q2 BIM Detector**: https://huggingface.co/DuckyDuck123/resnet34-bim-detector

---

Last updated: April 4, 2026
