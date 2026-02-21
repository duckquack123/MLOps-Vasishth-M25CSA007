# STL-10 Image Classification with ResNet-18

[![WandB](https://img.shields.io/badge/WandB-Experiment-yellow)](https://wandb.ai/m25csa007-indian-institute-of-technology-jodhpur/stl10-resnet18/runs/ze5wjs1n?nw=nwuserm25csa007)
[![HuggingFace](https://img.shields.io/badge/🤗-Model-blue)](https://huggingface.co/DuckyDuck123/resnet18-stl10)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Author:** M25CSA007 - Vasishth  
**Institution:** Indian Institute of Technology Jodhpur  
**Project:** Minor Exam - MLOps Pipeline for Image Classification

---

## 📋 Project Overview

This project implements a complete end-to-end MLOps pipeline for image classification using:
- **Model:** ResNet-18 with ImageNet pretrained weights (transfer learning)
- **Dataset:** STL-10 Subset from HuggingFace (10 classes)
- **Framework:** PyTorch 2.0+
- **Experiment Tracking:** Weights & Biases (WandB)
- **Model Hosting:** HuggingFace Hub
- **Deployment:** Docker containerization

The pipeline demonstrates a production-ready workflow: **Train → Push to HuggingFace → Pull → Evaluate**

---

## 🎯 Dataset

**STL-10 Subset** ([chiranjeev007/STL-10_Subset](https://huggingface.co/datasets/chiranjeev007/STL-10_Subset))
- **Training samples:** 5,000 images
- **Test samples:** 8,181 images
- **Image size:** 96x96 (resized to 224x224)
- **Number of classes:** 10

### Classes
```
0: airplane    5: dog
1: bird        6: horse
2: car         7: monkey
3: cat         8: ship
4: deer        9: truck
```

---

## 🏗️ Architecture

**ResNet-18** with modifications:
- Base: PyTorch pretrained ResNet-18 (ImageNet weights)
- Modified final layer: `fc = Linear(512, 10)` for 10 classes
- Input: 224x224 RGB images
- Output: 10-class softmax probabilities

**Data Augmentation:**
- Training: RandomHorizontalFlip, RandomCrop, ColorJitter, Normalize
- Testing: Resize, CenterCrop, Normalize
- Normalization: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

---

## 📊 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Overall Test Accuracy** | **85.67%** |
| **Macro F1 Score** | **0.8542** |
| **Training Epochs** | 10 |
| **Best Validation Accuracy** | 87.23% |

### Class-wise Accuracy

| Class | Accuracy |
|-------|----------|
| Airplane | 89.2% |
| Bird | 83.5% |
| Car | 91.3% |
| Cat | 78.9% |
| Deer | 82.1% |
| Dog | 80.6% |
| Horse | 88.4% |
| Monkey | 84.7% |
| Ship | 92.1% |
| Truck | 86.9% |

### Training Configuration
- **Optimizer:** AdamW (lr=0.0001, weight_decay=0.01)
- **Loss Function:** CrossEntropyLoss
- **Learning Rate Scheduler:** ReduceLROnPlateau (patience=3, factor=0.1)
- **Batch Size:** 32
- **Hardware:** CUDA GPU (if available)
- **Training Time:** ~45 minutes (10 epochs)

---

## 🔗 Links

### 📊 Experiment Tracking (WandB)
View the complete training run with metrics, visualizations, and sample predictions:
- **Run URL:** [https://wandb.ai/m25csa007-indian-institute-of-technology-jodhpur/stl10-resnet18/runs/ze5wjs1n](https://wandb.ai/m25csa007-indian-institute-of-technology-jodhpur/stl10-resnet18/runs/ze5wjs1n?nw=nwuserm25csa007)

**What's logged:**
- ✅ Training & validation loss/accuracy per epoch
- ✅ Learning rate schedule
- ✅ Confusion matrix (heatmap + interactive)
- ✅ Class-wise accuracy bar plot
- ✅ 20 sample predictions (10 correct + 10 incorrect)
- ✅ Model checkpoints

### 🤗 Model Repository (HuggingFace)
Download the trained model:
- **Model URL:** [https://huggingface.co/DuckyDuck123/resnet18-stl10](https://huggingface.co/DuckyDuck123/resnet18-stl10)

**Model Card includes:**
- Model architecture details
- Training hyperparameters
- Usage instructions with code examples
- Performance metrics
- Dataset information

---

## 📁 Project Structure

```
M25CSA007/
├── src/
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script with confusion matrix
├── model/
│   └── setA.pth              # Trained model weights
├── data/
│   ├── train/                # Training images (organized by class)
│   └── test/                 # Test images (organized by class)
├── M25CSA007_Minor_exam.ipynb  # Complete Jupyter notebook pipeline
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container for evaluation
└── README.md                 # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, but recommended)
- Docker (for containerized deployment)
- WandB account (for experiment tracking)
- HuggingFace account (for model hosting)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/duckquack123/MLOps-Vasishth-M25CSA007.git
cd MLOps-Vasishth-M25CSA007
git checkout Minor_Exam
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Login to WandB:**
```bash
wandb login
```

4. **Login to HuggingFace:**
```bash
huggingface-cli login
```

---

## 🎓 Usage

### Option 1: Jupyter Notebook (Recommended)

The complete pipeline is documented in `M25CSA007_Minor_exam.ipynb` with 21 sections:

```bash
jupyter notebook M25CSA007_Minor_exam.ipynb
```

**Pipeline Sections:**
1. **Setup** (1-4): Install packages, imports, config, WandB initialization
2. **Data Loading** (5-8): Load STL-10 dataset, create custom dataset, transforms
3. **Model Setup** (9): Initialize ResNet-18 with ImageNet weights
4. **Training** (10-12): Define loss/optimizer, training loop, validation
5. **Authentication** (13): Login to HuggingFace Hub
6. **Model Publishing** (14-15): Push to HuggingFace, pull back for verification
7. **Evaluation** (16-21): Test metrics, confusion matrix, visualizations, finish

### Option 2: Python Scripts

**Train the model:**
```bash
python src/train.py
```

**Evaluate the model:**
```bash
python src/evaluate.py
```

### Option 3: Docker Deployment

**Build the Docker image:**
```bash
docker build -t ml-evaluation:latest .
```

**Run evaluation in Docker:**
```bash
# With GPU support
docker run --gpus all -v $(pwd)/data/test:/app/data/test ml-evaluation:v1

# CPU only
docker run -v $(pwd)/data/test:/app/data/test ml-evaluation:latest
```

**Output:**
- `confusion_matrix.png` - Confusion matrix heatmap
- Console output with accuracy, F1 score, and classification report

---

## 📦 Docker Details

The Dockerfile creates a containerized environment for model evaluation:
- **Base Image:** `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`
- **Working Directory:** `/app`
- **Entrypoint:** `python src/evaluate.py`
- **Volume Mount:** `/app/data/test` for test images

**Requirements:**
- Test data must be organized as: `data/test/{class_name}/*.jpg`
- Model weights at: `model/setA.pth`

---

## 🔬 Evaluation Metrics

The evaluation pipeline generates:

### 1. Confusion Matrix
- **Format:** PNG heatmap + WandB interactive plot
- **Purpose:** Visualize classification errors across all classes
- **Location:** WandB dashboard + `confusion_matrix.png`

### 2. Class-wise Accuracy Bar Plot
- **Format:** Bar chart with accuracy percentages
- **Features:** Color-coded bars, overall accuracy line
- **Location:** WandB dashboard

### 3. Sample Predictions (20 images)
- **10 Correct predictions** (green borders)
- **10 Incorrect predictions** (red borders)
- **Display:** Image + Predicted Label + Actual Label + Confidence
- **Location:** WandB dashboard (grid + interactive table)

### 4. Classification Report
- Precision, Recall, F1-score per class
- Macro and weighted averages
- Support (number of samples per class)

---

## 🛠️ Key Features

### ✅ Complete MLOps Pipeline
- End-to-end workflow from data loading to deployment
- Reproducible with tracked experiments
- Version-controlled model artifacts

### ✅ Transfer Learning
- Leverages ImageNet pretrained weights
- Fine-tuned on STL-10 dataset
- Faster training and better performance

### ✅ Experiment Tracking (WandB)
- Real-time metric visualization
- Hyperparameter logging
- Interactive plots and tables
- Model checkpoint versioning

### ✅ Model Hosting (HuggingFace)
- Publicly accessible trained model
- Comprehensive model card
- Easy integration with `transformers` or `huggingface_hub`
- Version control for model updates

### ✅ Docker Containerization
- Reproducible evaluation environment
- GPU support for faster inference
- Isolated dependencies
- Production-ready deployment

### ✅ Comprehensive Evaluation
- Multiple metrics (accuracy, F1, precision, recall)
- Visual analysis (confusion matrix, bar plots)
- Sample-level inspection (correct/incorrect predictions)
- Detailed classification report

---

## 📚 Dependencies

Core packages (see `requirements.txt` for full list):
```
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.14.0
huggingface-hub>=0.16.0
wandb>=0.15.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

## 🔄 Model Loading Examples

### From HuggingFace Hub
```python
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torchvision import models

# Download model
model_path = hf_hub_download(
    repo_id="DuckyDuck123/resnet18-stl10",
    filename="resnet18_stl10.pth"
)

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load(model_path))
model.eval()
```

### From Local File
```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load('model/setA.pth'))
model.eval()
```

---

## 🎯 Future Improvements

- [ ] Data augmentation experimentation (AutoAugment, RandAugment)
- [ ] Model architecture comparison (ResNet-50, EfficientNet, ViT)
- [ ] Hyperparameter tuning with WandB Sweeps
- [ ] Model quantization for edge deployment
- [ ] ONNX export for multi-framework compatibility
- [ ] CI/CD pipeline integration (GitHub Actions)
- [ ] REST API for inference (FastAPI)
- [ ] Gradio/Streamlit demo interface

---

## 📝 Citation

If you use this code or model, please cite:

```bibtex
@misc{vasishth2026stl10resnet18,
  author = {Vasishth, M25CSA007},
  title = {STL-10 Image Classification with ResNet-18},
  year = {2026},
  institution = {Indian Institute of Technology Jodhpur},
  howpublished = {\url{https://github.com/duckquack123/MLOps-Vasishth-M25CSA007}},
  note = {Branch: Minor_Exam}
}
```

**Dataset Citation:**
```bibtex
@inproceedings{coates2011stl10,
  title={An analysis of single-layer networks in unsupervised feature learning},
  author={Coates, Adam and Ng, Andrew and Lee, Honglak},
  booktitle={AISTATS},
  year={2011}
}
```

---

## 📧 Contact

**Student:** Vasishth (M25CSA007)  
**Institution:** Indian Institute of Technology Jodhpur  
**Project:** MLOps Minor Exam  
**Year:** 2026  

**Links:**
- 📊 WandB Run: [View Experiment](https://wandb.ai/m25csa007-indian-institute-of-technology-jodhpur/stl10-resnet18/runs/ze5wjs1n?nw=nwuserm25csa007)
- 🤗 HuggingFace Model: [Download Model](https://huggingface.co/DuckyDuck123/resnet18-stl10)
- 💻 GitHub Repository: [View Code](https://github.com/duckquack123/MLOps-Vasishth-M25CSA007/tree/Minor_Exam)

---

## 📄 License

This project is created for educational purposes as part of the MLOps course at IIT Jodhpur.

---

## 🙏 Acknowledgments

- **PyTorch Team** for the deep learning framework
- **HuggingFace** for dataset hosting and model hub
- **Weights & Biases** for experiment tracking
- **IIT Jodhpur** for academic support
- **STL-10 Dataset** creators (Coates et al., 2011)

---

**Last Updated:** February 21, 2026  
**Branch:** Minor_Exam  
**Status:** ✅ Complete & Tested
