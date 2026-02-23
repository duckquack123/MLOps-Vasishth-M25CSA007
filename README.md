# MLOps Assignment 3: Fine-Tuning BERT for Sentiment Classification

This project demonstrates fine-tuning a BERT model (bert-tiny) on the IMDB dataset for sentiment classification using Hugging Face Transformers. The project includes training, evaluation, and deployment using Docker containers.

## 📋 Project Overview

- **Model**: `prajjwal1/bert-tiny` (fine-tuned version available at `DuckyDuck123/bert-tiny-imdb`)
- **Dataset**: IMDB Movie Reviews (sentiment classification)
- **Framework**: PyTorch, Hugging Face Transformers
- **Task**: Binary sentiment classification (positive/negative)

## 🚀 Features

- Fine-tuning BERT model for text classification
- Automated training pipeline with checkpointing
- Comprehensive model evaluation with metrics
- Docker containerization for training and evaluation
- Model versioning and Hugging Face Hub integration

## 📁 Project Structure

```
.
├── src/
│   ├── train.py              # Training script
│   ├── evaluate_model.py     # Evaluation script
│   └── data.py              # Data loading and preprocessing
├── models/                   # Saved model checkpoints
├── results/                 # Evaluation results
├── notebooks/               # Jupyter notebooks for experimentation
├── Dockerfile.train         # Docker image for training
├── Dockerfile.eval          # Docker image for evaluation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/duckquack123/MLOps-Vasishth-M25CSA007.git
cd MLOps-Vasishth-M25CSA007
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup

#### Training
```bash
docker build -f Dockerfile.train -t bert-train .
docker run --gpus all bert-train
```

#### Evaluation
```bash
docker build -f Dockerfile.eval -t bert-eval .
docker run bert-eval
```

## 📊 Usage

### Training

```bash
python src/train.py
```

This will:
- Load the IMDB dataset
- Tokenize the text data
- Fine-tune the BERT-tiny model
- Save checkpoints to `models/`
- Push the final model to Hugging Face Hub

### Evaluation

```bash
python src/evaluate_model.py
```

This will:
- Load the fine-tuned model from Hugging Face Hub
- Evaluate on the test set
- Generate metrics (accuracy, F1, precision, recall)
- Save results to `results/`

### Data Preparation

```python
from src.data import load_and_prepare_data

train_dataset, test_dataset, tokenizer = load_and_prepare_data()
```

## 📈 Model Performance

The fine-tuned model achieves the following metrics on the IMDB test set:
- Accuracy: ~XX%
- F1 Score: ~XX%
- Precision: ~XX%
- Recall: ~XX%

(Check `results/eval_results.json` for detailed metrics)

## 🔧 Configuration

Key configurations in `src/train.py`:
- `MODEL_NAME`: Base model from Hugging Face
- `OUTPUT_DIR`: Directory to save checkpoints
- `HF_REPO`: Hugging Face Hub repository name
- Training parameters: batch size, learning rate, epochs, etc.

## 📦 Requirements

- Python 3.10+
- PyTorch 2.3.0
- Transformers 4.41.2
- Datasets 2.19.1
- CUDA 12.1+ (for GPU training)

See `requirements.txt` for complete list of dependencies.

## 🐳 Docker Images

The project includes two Docker images:

1. **Training Image** (`Dockerfile.train`): CUDA-enabled for GPU training
2. **Evaluation Image** (`Dockerfile.eval`): Lightweight CPU-only image

## 📝 Results

Training and evaluation results are stored in:
- `results/eval_results.json`: Evaluation metrics
- `results/hf_eval_results.json`: Hugging Face evaluation results
- `models/`: Model checkpoints and configurations

## 🤝 Contributing

This is an academic assignment project. For any questions or issues, please contact the repository owner.

## 📄 License

This project is created for educational purposes as part of MLOps coursework.

## 👤 Author

**Vasishth** (M25CSA007)

## 🔗 Links

- Hugging Face Model: [DuckyDuck123/bert-tiny-imdb](https://huggingface.co/DuckyDuck123/bert-tiny-imdb)
- Base Model: [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)
- Dataset: [IMDB Movie Reviews](https://huggingface.co/datasets/imdb)

## 📚 References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [IMDB Dataset Paper](https://ai.stanford.edu/~amaas/data/sentiment/)
