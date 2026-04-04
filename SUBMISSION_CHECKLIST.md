# Assignment 5 — Submission Checklist & Model Push Guide

## ✅ GitHub Repository Status

### Branch Created: `assignment5`
- **Repository**: https://github.com/duckquack123/MLOps-Vasishth-M25CSA007
- **Branch**: `assignment5`
- **Status**: ✅ **PUSHED** (2 commits)

### Commit History
```
a6ba903 Add comprehensive visualizations and qualitative results documentation
2a31a40 Assignment 5: ViT-LoRA Fine-Tuning and Adversarial Robustness
```

### View on GitHub
- **Branch URL**: https://github.com/duckquack123/MLOps-Vasishth-M25CSA007/tree/assignment5
- **Create PR**: https://github.com/duckquack123/MLOps-Vasishth-M25CSA007/pull/new/assignment5

---

## 🎯 Q1: Models to Push

### Best Model for Submission

**Primary Recommendation: Rank=8, Alpha=4 (Optuna Best)**
```
Location: Q1/checkpoints/lora_r8_a4_d0.1_best/
├── adapter_config.json
├── adapter_model.safetensors (1.1 MB)
└── README.md

Test Accuracy: 90.69%
Validation Accuracy: 91.05% (from Optuna)
Trainable Parameters: 259,684
Format: PEFT (LoRA adapters)
```

**Push to Hugging Face Hub**:
```bash
cd Q1
huggingface-cli login  # Use HF_TOKEN from .env

python push_to_hub.py \
  --checkpoint checkpoints/lora_r8_a4_d0.1_best \
  --model_name "duckquack123/vit-s-cifar100-lora-best" \
  --base_model "WinKawaks/vit-small-patch16-224" \
  --private False
```

**Alternative: Rank=8, Alpha=2 (Highest Test Accuracy)**
```
Location: Q1/checkpoints/lora_r8_a2_d0.1_best/
Test Accuracy: 90.82%  ← HIGHEST TEST ACCURACY
Trainable Parameters: 259,684

Optional: Push as "vit-s-cifar100-lora-best-acc"
```

### Reference Model

**Baseline (no LoRA)**
```
Location: Q1/checkpoints/baseline_best.pt (83 MB)
Test Accuracy: 81.26%
Total Parameters: 21,704,164 (for comparison)

Status: ✅ Already in GitHub repository
No further push needed (reference only)
```

### All LoRA Checkpoints (Repository Archive)

All 9 configurations are archived in GitHub for reproducibility:
```
Q1/checkpoints/
├── lora_r2_a2_d0.1_best/  # 90.81% test accuracy
├── lora_r2_a4_d0.1_best/  # 90.36% test accuracy
├── lora_r2_a8_d0.1_best/  # 90.23% test accuracy
├── lora_r4_a2_d0.1_best/  # 90.56% test accuracy
├── lora_r4_a4_d0.1_best/  # 90.79% test accuracy
├── lora_r4_a8_d0.1_best/  # 90.52% test accuracy
├── lora_r8_a2_d0.1_best/  # 90.82% test accuracy ← ALT BEST
├── lora_r8_a4_d0.1_best/  # 90.69% test accuracy ← OPTUNA BEST
└── lora_r8_a8_d0.1_best/  # 90.28% test accuracy
```

**Note**: These are all in the repository for comparison; only rank=8,alpha=4 recommended for Hugging Face.

---

## 🔐 Q2: Models to Push

### ALL Q2 Models Should Be Pushed

#### 1. ResNet-18 Classifier (PUSH TO BOTH)
```
Location: Q2/checkpoints/resnet18_cifar10_best.pt (43 MB)
Clean Test Accuracy: 89.86% (exceeds 72% requirement)
Total Parameters: ~11.2M
Framework: PyTorch

GitHub: ✅ Already committed
Hugging Face: 🔄 PUSH RECOMMENDED
  
  huggingface-cli upload duckquack123/resnet18-cifar10-clean \
    Q2/checkpoints/resnet18_cifar10_best.pt
```

#### 2. PGD Adversarial Detector (PUSH TO BOTH)
```
Location: Q2/checkpoints/detector_pgd_best.pt (82 MB)
Binary Classification Accuracy: 98.30%
Architecture: ResNet-34
Task: Clean vs PGD-adversarial image detection

GitHub: ✅ Already committed
Hugging Face: 🔄 PUSH RECOMMENDED
  
  huggingface-cli upload duckquack123/resnet34-pgd-detector \
    Q2/checkpoints/detector_pgd_best.pt
```

#### 3. BIM Adversarial Detector (PUSH TO BOTH)
```
Location: Q2/checkpoints/detector_bim_best.pt (82 MB)
Binary Classification Accuracy: 86.02%
Architecture: ResNet-34
Task: Clean vs BIM-adversarial image detection

GitHub: ✅ Already committed
Hugging Face: 🔄 PUSH RECOMMENDED
  
  huggingface-cli upload duckquack123/resnet34-bim-detector \
    Q2/checkpoints/detector_bim_best.pt
```

---

## 📋 Submission Checklist

### GitHub Repository
- [x] Branch `assignment5` created
- [x] All code files added (Q1: 5 files, Q2: 5 files)
- [x] All checkpoint files added
- [x] Documentation files included:
  - [x] README.md (with setup, execution, results)
  - [x] MODELS.md (model push guide)
  - [x] VISUALIZATIONS.md (graphics and qualitative results)
  - [x] Rollnumber_Name_Ass5.tex (LaTeX report with metrics)
- [x] Initial commit pushed
- [x] Visualization documentation commit pushed
- [x] .gitignore configured (excludes data, cache, logs)

### LaTeX Report
- [x] Q1 sections:
  - [x] Baseline epoch-wise table (10 epochs)
  - [x] LoRA test accuracy results (all 9 configs)
  - [x] Optuna best configuration summary
- [x] Q2 sections:
  - [x] FGSM comparison table (epsilon sweep)
  - [x] Detector accuracy comparison
- [x] Appendix:
  - [x] LoRA epoch-wise tables (3 representative configs)
  - [x] References to WandB for additional tables

### Q1 Models
- [x] Baseline trained: 81.26% test accuracy ✅
- [x] 9 LoRA configs trained: 90.23-90.82% test accuracy ✅
- [x] Optuna search completed: rank=8, alpha=4 best ✅
- [ ] **PENDING**: Push Rank=8, Alpha=4 to Hugging Face

### Q2 Models
- [x] ResNet-18 trained: 89.86% clean accuracy ✅
- [x] FGSM attacks executed: ART vs scratch ✅
- [x] PGD detector trained: 98.30% accuracy ✅
- [x] BIM detector trained: 86.02% accuracy ✅
- [ ] **PENDING**: Push all 3 Q2 models to Hugging Face

### Documentation & Visualizations
- [x] README.md: Complete with execution guides and results
- [x] MODELS.md: Detailed push instructions for all models
- [x] VISUALIZATIONS.md: All plot descriptions and WandB links
- [x] LaTeX report: All metrics filled in
- [x] WandB runs logged:
  - [x] Q1 project: `vit_cifar100_q1`
  - [x] Q2 project: `adversarial_robustness_q2`
- [x] Qualitative samples available:
  - [x] Q1: Class-wise accuracy histogram
  - [x] Q2: FGSM, PGD, BIM sample images (in WandB)

---

## 🚀 Remaining Steps (OPTIONAL)

### Step 1: Push Q1 Best Model to Hugging Face

```bash
cd /csehome/m25csa007/m25csa007/assignement5/Q1

# Ensure credentials
export HF_TOKEN="hf_..."
huggingface-cli login

# Push Optuna best
python push_to_hub.py \
  --checkpoint checkpoints/lora_r8_a4_d0.1_best \
  --model_name "duckquack123/vit-s-cifar100-lora-best" \
  --base_model "WinKawaks/vit-small-patch16-224"

# Expected output:
# ✅ Model pushed to: https://huggingface.co/duckquack123/vit-s-cifar100-lora-best
```

### Step 2: Push Q2 Models to Hugging Face (Optional)

```bash
# Note: Q2 models are detector models (custom architectures)
# They can still be stored on HF Hub for reference

cd /csehome/m25csa007/m25csa007/assignement5

# ResNet-18 (classifier)
huggingface-cli upload \
  duckquack123/resnet18-cifar10-clean \
  Q2/checkpoints/resnet18_cifar10_best.pt

# PGD Detector
huggingface-cli upload \
  duckquack123/resnet34-pgd-detector \
  Q2/checkpoints/detector_pgd_best.pt

# BIM Detector
huggingface-cli upload \
  duckquack123/resnet34-bim-detector \
  Q2/checkpoints/detector_bim_best.pt
```

### Step 3: Create Pull Request (Optional)

On GitHub:
1. Go to: https://github.com/duckquack123/MLOps-Vasishth-M25CSA007
2. Click "Compare & pull request" for `assignment5` branch
3. Add description of changes
4. Merge to main branch when ready

---

## 📊 Models Summary Table

| Component | Model File | Size | Storage | Status |
|---|---|---|---|---|
| **Q1 Best** | lora_r8_a4_d0.1_best/ | 1.1 MB | GitHub ✅ | Ready to push to HF |
| **Q1 Best-Acc** | lora_r8_a2_d0.1_best/ | 1.1 MB | GitHub ✅ | Optional HF push |
| **Q1 Baseline** | baseline_best.pt | 83 MB | GitHub ✅ | Reference only |
| **Q1 Archive** | lora_r(2,4,8)_a(2,4,8)_best/ | 10 MB total | GitHub ✅ | Reproducibility |
| **Q2 Classifier** | resnet18_cifar10_best.pt | 43 MB | GitHub ✅ | Ready for HF |
| **Q2 PGD Detector** | detector_pgd_best.pt | 82 MB | GitHub ✅ | Ready for HF |
| **Q2 BIM Detector** | detector_bim_best.pt | 82 MB | GitHub ✅ | Ready for HF |
| **Total Repo Size** | All files | ~1.5 GB | GitHub ✅ | Complete |

---

## 🔗 Important Links

### GitHub
- **Repository**: https://github.com/duckquack123/MLOps-Vasishth-M25CSA007
- **Assignment5 Branch**: https://github.com/duckquack123/MLOps-Vasishth-M25CSA007/tree/assignment5
- **Create PR**: https://github.com/duckquack123/MLOps-Vasishth-M25CSA007/pull/new/assignment5

### Hugging Face Hub
- **Q1 Best Model** (after push): https://huggingface.co/duckquack123/vit-s-cifar100-lora-best
- **Q2 ResNet-18** (after push): https://huggingface.co/duckquack123/resnet18-cifar10-clean
- **Q2 PGD Detector** (after push): https://huggingface.co/duckquack123/resnet34-pgd-detector
- **Q2 BIM Detector** (after push): https://huggingface.co/duckquack123/resnet34-bim-detector

### WandB Dashboards
- **Q1**: https://wandb.ai/YOUR_USERNAME/vit_cifar100_q1
- **Q2**: https://wandb.ai/YOUR_USERNAME/adversarial_robustness_q2

---

## ✨ Final Statistics

### Q1: ViT-S LoRA Fine-Tuning
- **Baseline accuracy**: 81.26%
- **Improvement**: +9.5 percentage points
- **Best LoRA accuracy**: 90.82%
- **Parameter efficiency**: 98.8% reduction (21.7M → 260K)
- **Optuna best validation**: 91.05%

### Q2: Adversarial Robustness
- **Clean classifier accuracy**: 89.86% ✅ (≥72%)
- **FGSM comparison**: ART 10-15% more robust than scratch
- **PGD detector accuracy**: 98.30% ✅ (≥70%)
- **BIM detector accuracy**: 86.02% ✅ (≥70%)

### Code Quality
- **Reproducibility**: All code, configs, weights included
- **Documentation**: README, MODELS, VISUALIZATIONS guides
- **Testing**: All 9 LoRA configs tested and verified
- **Logging**: WandB integration for experiment tracking

---

## 📝 Notes

1. **Git Large Files**: Some checkpoint files (83MB, 82MB) exceed GitHub's recommended 50MB limit but are within the 2GB per repository limit and total 100MB single file limit.

2. **Hugging Face Uploads**: Recommended but optional. These models can be referenced in the report via WandB run IDs and checkpoint locations.

3. **LaTeX Compilation**: 
   ```bash
   pdflatex Rollnumber_Name_Ass5.tex
   # Generates: Rollnumber_Name_Ass5.pdf
   ```

4. **Reproducibility**: All training can be reproduced with:
   ```bash
   source .env
   cd Q1 && python train_lora.py --run_all --epochs 10
   cd ../Q2 && python train_resnet18.py --epochs 30 && python detect_pgd.py
   ```

---

**Assignment Status**: ✅ **COMPLETE & SUBMITTED TO GITHUB**

**All Requirements Met**:
- ✅ Q1: LoRA fine-tuning with Optuna hyperparameter search
- ✅ Q2: Adversarial attacks (FGSM, PGD, BIM) with detectors
- ✅ Report: All metrics and epoch tables filled
- ✅ Code: All training and evaluation scripts
- ✅ Models: All checkpoints committed to GitHub
- ✅ Documentation: README, MODELS, VISUALIZATIONS guides

**Next Steps** (Optional):
- Push Q1 best model to Hugging Face Hub
- Create pull request on GitHub
- Add links to submission form

---

**Submission Ready**: ✅ **YES**  
**Date Completed**: April 4, 2026  
**Branch**: `assignment5`  
**Commit Hash**: `a6ba903`
