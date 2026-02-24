# MLOps Assignment 3: Fine-Tuning DistilBERT for Book Genre Classification

This project demonstrates an end-to-end machine learning workflow - from notebook to production - featuring fine-tuning of a DistilBERT model on the Goodreads book reviews dataset for multi-class genre classification. The project includes training, evaluation, model versioning via Hugging Face Hub, and Docker containerization.

## 📋 Project Overview

- **Model**: `distilbert-base-cased` → Fine-tuned: `DuckyDuck123/distilbert-goodreads-genre`
- **Dataset**: [Goodreads Book Reviews](https://mengtingwan.github.io/data/goodreads.html#datasets) (UCSD)
- **Framework**: PyTorch, Hugging Face Transformers
- **Task**: Multi-class genre classification
- **Classes**: 8 genres (Poetry, Children, Comics/Graphic, Fantasy/Paranormal, History/Biography, Mystery/Thriller/Crime, Romance, Young Adult)
- **Training Data**: 6,400 samples (800 per genre)
- **Test Data**: 1,600 samples (200 per genre)

## 🎯 Assignment Objectives Completed

✅ **Task 1-3**: Notebook converted to production Python scripts  
✅ **Task 4**: Pre-trained DistilBERT model from Hugging Face  
✅ **Task 5**: Model training using Trainer API  
✅ **Task 6**: Comprehensive evaluation with metrics  
✅ **Task 7**: Model uploaded to Hugging Face Hub  
✅ **Task 8**: Re-evaluation from Hugging Face repository  
✅ **Task 9**: Production Docker image for evaluation  
✅ **Task 10**: Complete codebase on GitHub

## 📚 Dataset

The project uses the [Goodreads book reviews dataset](https://mengtingwan.github.io/data/goodreads.html#datasets) which includes reviews across 8 different genres:

1. Poetry
2. Children
3. Comics & Graphic
4. Fantasy & Paranormal
5. History & Biography
6. Mystery, Thriller & Crime
7. Romance
8. Young Adult

Each genre has reviews sampled from the full dataset, with default configuration of 800 training and 200 test samples per genre.

## 🚀 Key Features

- ✨ **Production-Ready Pipeline**: Notebook converted to modular Python scripts
- 🤖 **State-of-the-Art Model**: DistilBERT fine-tuned for 8-genre classification
- 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 (macro-averaged)
- 🔄 **Automated Data Pipeline**: Streams and caches Goodreads reviews
- 💾 **Model Versioning**: Integrated with Hugging Face Hub
- 🐳 **Docker Support**: Containerized training and evaluation
- ⚡ **GPU Acceleration**: Automatic CUDA detection and usage
- 📈 **Detailed Reporting**: Per-genre classification reports

## 📁 Project Structure

```
hf-assignment/
├── src/
│   ├── data.py                 # Data loading & preprocessing
│   ├── train.py                # Training pipeline with HF Hub upload
│   └── evaluate_model.py       # Evaluation from HF Hub
├── models/                     # Trained model artifacts
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── config_labels.json
├── results/                    # Evaluation results
│   ├── eval_results.json       # Training evaluation metrics
│   ├── classification_report.json  # Per-genre metrics
│   └── hf_eval_results.json    # HuggingFace model evaluation
├── data/                       # Cached datasets
│   ├── genre_reviews_dict.pickle
│   └── label_encoder.pickle
├── notebooks/
│   └── ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb
├── logs/                       # Training logs
├── Dockerfile.train            # Training container (GPU)
├── Dockerfile.eval             # Evaluation container (CPU)
├── upload_to_hf.py            # HuggingFace upload utility
├── requirements.txt
└── README.md
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.3.0
- CUDA 12.1+ (optional, for GPU acceleration)
- Docker (for containerized deployment)
- HuggingFace account (for model upload)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/duckquack123/MLOps-Vasishth-M25CSA007.git
   cd MLOps-Vasishth-M25CSA007
   git checkout assignment-3
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU availability (optional):**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Docker Setup

**Build Docker images:**

```bash
# Training image (with GPU support)
docker build -f Dockerfile.train -t distilbert-train .

# Evaluation image
docker build -f Dockerfile.eval -t distilbert-eval .
```

## 📊 Usage

### 1. Training the Model

**Local training:**

```bash
python src/train.py
```

**What it does:**
- ✓ Auto-detects GPU/CPU and displays hardware info
- ✓ Downloads and caches Goodreads dataset (first run)
- ✓ Fine-tunes DistilBERT for 3 epochs
- ✓ Evaluates model with comprehensive metrics
- ✓ Saves to `models/` directory
- ✓ Generates detailed classification report
- ✓ Prompts to upload model to Hugging Face Hub

**Output files:**
- `models/` - Trained model artifacts
- `results/eval_results.json` - Overall metrics
- `results/classification_report.json` - Per-genre performance
- `data/` - Cached dataset and label encoder

**Expected training time:**
- GPU (NVIDIA RTX 3080): ~15-20 minutes
- CPU: ~2-3 hours

**Docker training:**
```bash
docker run --gpus all -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results distilbert-train
```

### 2. Uploading Model to Hugging Face

**Option A: During training** (when prompted, enter `yes`)

**Option B: After training with upload script:**

```bash
python upload_to_hf.py
```

You'll need your HuggingFace token from: https://huggingface.co/settings/tokens

### 3. Evaluating from Hugging Face Hub

**Local evaluation:**

```bash
python src/evaluate_model.py
```

**What it does:**
- ✓ Downloads model from `DuckyDuck123/distilbert-goodreads-genre`
- ✓ Falls back to local `models/` if HF unavailable
- ✓ Runs evaluation on test set
- ✓ Generates comprehensive metrics
- ✓ Saves results to `results/hf_eval_results.json`

**Docker evaluation:**
```bash
docker run --rm distilbert-eval
```

This automatically:
- Pulls model from HuggingFace Hub
- Runs evaluation
- Displays results in container logs

## 📈 Model Performance

The fine-tuned DistilBERT model is evaluated on 1,600 test samples (200 per genre) with the following metrics:

**Evaluation Metrics:**
- **Accuracy**: Overall classification accuracy across all genres
- **Precision (Macro)**: Average precision across all genres
- **Recall (Macro)**: Average recall across all genres  
- **F1 Score (Macro)**: Harmonic mean of precision and recall

**Genre-wise Performance:**

The model generates detailed per-genre classification reports including:
- Precision, Recall, F1-score for each of the 8 genres
- Support (number of samples) per genre
- Confusion patterns between similar genres

**Results Location:**
- `results/eval_results.json` - Overall metrics from training
- `results/classification_report.json` - Detailed per-genre breakdown
- `results/hf_eval_results.json` - Re-evaluation from HuggingFace Hub

**Example output:**
```json
{
    "eval_loss": 0.XXX,
    "eval_accuracy": 0.XX,
    "eval_precision": 0.XX,
    "eval_recall": 0.XX,
    "eval_f1": 0.XX,
    "eval_runtime": XX.XX,
    "eval_samples_per_second": XXX.XX
}
```

## 📚 Dataset Details

**Source**: [UCSD Goodreads Dataset](https://mengtingwan.github.io/data/goodreads.html#datasets)

**Genres Included:**
1. **Poetry** - Poetry collections and individual poems
2. **Children** - Children's literature and picture books
3. **Comics & Graphic** - Graphic novels and comic books
4. **Fantasy & Paranormal** - Fantasy fiction and paranormal romance
5. **History & Biography** - Historical accounts and biographies
6. **Mystery, Thriller & Crime** - Mystery and crime fiction
7. **Romance** - Romantic fiction
8. **Young Adult** - Young adult literature

**Data Processing:**
- Reviews are streamed from gzipped JSON files
- 10,000 reviews read per genre (configurable)
- 2,000 reviews randomly sampled per genre
- 80/20 train-test split (800/200 per genre)
- Cached locally after first download
- Tokenized with DistilBERT tokenizer (max length: 512)

## 🔧 Configuration

### Training Hyperparameters

**Model Configuration:**
```python
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512  # Maximum token sequence length
NUM_LABELS = 8    # Number of genre classes
```

**Training Arguments:**
```python
LEARNING_RATE = 2e-5
BATCH_SIZE = 8 (per device)
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
OPTIMIZER = AdamW (default)
LR_SCHEDULER = Linear with warmup
```

**Data Configuration:**
```python
TRAIN_SIZE = 800  # Samples per genre for training
TEST_SIZE = 200   # Samples per genre for testing
HEAD = 10000      # Reviews to read per genre
SAMPLE_SIZE = 2000  # Reviews to sample per genre
```

**Key Files:**
- `src/train.py` - Main training script
- `src/evaluate_model.py` - Evaluation script  
- `src/data.py` - Data loading and preprocessing
- `upload_to_hf.py` - HuggingFace upload utility

## 🐳 Docker Deployment

### Training Container (GPU-enabled)

**Dockerfile.train** - Based on `nvidia/cuda:12.1.1-runtime-ubuntu22.04`

```bash
# Build
docker build -f Dockerfile.train -t distilbert-train .

# Run with GPU
docker run --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  distilbert-train
```

### Evaluation Container (CPU)

**Dockerfile.eval** - Based on `python:3.10-slim`

```bash
# Build
docker build -f Dockerfile.eval -t distilbert-eval .

# Run
docker run --rm \
  -v $(pwd)/results:/app/results \
  distilbert-eval
```

**Note**: Evaluation container automatically pulls the model from HuggingFace Hub.

## 📦 Dependencies

**Core Libraries:**
- `torch==2.3.0` - Deep learning framework
- `transformers==4.41.2` - HuggingFace transformers
- `datasets==2.19.1` - Dataset utilities
- `evaluate==0.4.2` - Evaluation metrics
- `accelerate==0.30.1` - Training acceleration

**Data Processing:**
- `scikit-learn` - Label encoding, metrics
- `pandas` - Data manipulation
- `numpy==1.26.4` - Numerical operations
- `requests` - HTTP requests for data download

**Utilities:**
- `huggingface_hub==0.23.4` - HF Hub integration
- `sentencepiece` - Tokenization
- `tqdm` - Progress bars

See [`requirements.txt`](requirements.txt) for complete list.

## � Model Selection Rationale

**Why DistilBERT?**

1. **Efficiency**: 40% smaller and 60% faster than BERT-base while retaining 97% of its language understanding
2. **Performance**: Strong baseline for text classification tasks
3. **Resource-Friendly**: Suitable for both GPU and CPU deployment
4. **Well-Documented**: Extensive community support and documentation
5. **Production-Ready**: Optimized for inference with minimal performance degradation

**Comparison to alternatives:**
- **BERT-base**: More accurate but slower and larger
- **BERT-tiny**: Faster but significant performance drop
- **RoBERTa**: Better performance but much larger model size
- **DistilBERT**: Optimal balance of speed, size, and accuracy

## 🔍 GPU Acceleration

The training script automatically detects and uses GPU when available:

**GPU Detection Output:**
```
==================================================
Hardware Configuration
==================================================
✓ GPU available: NVIDIA GeForce RTX 3080
  Number of GPUs: 1
  CUDA Version: 11.8
==================================================

Loading model...
Model will use device: cuda
✓ Model will train on CUDA GPU

==================================================
Starting training...
✓ Training on GPU: NVIDIA GeForce RTX 3080
==================================================
```

**Monitor GPU usage:**
```bash
nvidia-smi -l 1  # Updates every second
```

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in src/train.py
per_device_train_batch_size=4  # Default is 8
```

**2. HuggingFace Hub Authentication**
```bash
# Get token from https://huggingface.co/settings/tokens
huggingface-cli login
# Or set environment variable
export HF_TOKEN="your_token_here"
```

**3. Dataset Download Issues**
```bash
# Clear cached data and retry
rm -rf data/genre_reviews_dict.pickle
python src/train.py
```

**4. Docker GPU Access**
```bash
# Install nvidia-docker2
# Then use --gpus all flag
docker run --gpus all distilbert-train
```

**5. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📝 Assignment Submission Checklist

- [x] **Notebook Downloaded**: ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb
- [x] **Environment Setup**: Docker files created
- [x] **Scripts Converted**: train.py, evaluate_model.py, data.py
- [x] **Model Selection**: DistilBERT-base-cased documented
- [x] **Training Complete**: Using HuggingFace Trainer API
- [x] **Evaluation Done**: Comprehensive metrics computed
- [x] **Model Uploaded**: Available on HuggingFace Hub
- [x] **Re-evaluation**: From HuggingFace repository
- [x] **Docker Image**: Evaluation container created
- [x] **GitHub Push**: Complete codebase available

## 🔗 Important Links

### Project Resources
- **GitHub Repository**: [duckquack123/MLOps-Vasishth-M25CSA007](https://github.com/duckquack123/MLOps-Vasishth-M25CSA007/tree/assignment-3)
- **HuggingFace Model**: [DuckyDuck123/distilbert-goodreads-genre](https://huggingface.co/DuckyDuck123/distilbert-goodreads-genre)
- **Base Model**: [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)

### Dataset & References
- **Goodreads Dataset**: [UCSD Book Review Data](https://mengtingwan.github.io/data/goodreads.html#datasets)
- **Dataset Paper**: [Wan & McAuley, RecSys 2018](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys17.pdf)

### Documentation
- **HuggingFace Transformers**: [Official Docs](https://huggingface.co/docs/transformers)
- **DistilBERT Paper**: [Sanh et al., 2019](https://arxiv.org/abs/1910.01108)
- **Trainer API**: [Training Guide](https://huggingface.co/docs/transformers/training)

## 📄 Project Report Summary

### Model Selection
Selected **DistilBERT-base-cased** for its optimal balance of performance and efficiency. The model provides strong text classification capabilities while being 40% smaller than BERT-base, making it suitable for both training and deployment.

### Training Summary
- **Dataset**: 6,400 training samples across 8 genres
- **Epochs**: 3
- **Batch Size**: 8 per device
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with linear warmup
- **Hardware**: CUDA-enabled GPU (when available)
- **Training Time**: ~15-20 minutes on NVIDIA RTX 3080

### Evaluation Comparison

| Metric | Local Evaluation | HuggingFace Re-evaluation |
|--------|------------------|---------------------------|
| Source | `results/eval_results.json` | `results/hf_eval_results.json` |
| Model | Locally saved | Downloaded from HF Hub |
| Purpose | Validate training | Verify model upload |

Both evaluations should produce identical results, confirming successful model upload and retrieval.

### Challenges Faced
1. **Dataset Size**: Goodreads dataset is large; implemented streaming and caching
2. **Multi-class Metrics**: Used macro-averaging for balanced genre representation
3. **Docker GPU**: Required nvidia-docker2 for GPU access in containers
4. **HuggingFace Upload**: Implemented both automatic and manual upload options

### Key Achievements
- ✅ Complete MLOps pipeline from notebook to production
- ✅ Modular, maintainable code structure
- ✅ Comprehensive evaluation and monitoring
- ✅ Containerized deployment
- ✅ Model version control via HuggingFace Hub

## 👤 Author

**Vasishth** (M25CSA007)  
MLOps Assignment 3 - Fine-Tuning & Deployment  
Course: ML/DL Operations

---

## 📄 License

This project is created for educational purposes as part of MLOps coursework (2026).

---

**Last Updated**: February 2026  
**Assignment**: MLOps Assignment 3 - End-to-End Model Training & Docker Deployment
