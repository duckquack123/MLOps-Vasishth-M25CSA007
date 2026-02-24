# MLOps Assignment 3: Fine-Tuning DistilBERT for Book Genre Classification

This project demonstrates fine-tuning a DistilBERT model on the Goodreads book reviews dataset for multi-class genre classification using Hugging Face Transformers. The project includes training, evaluation, and deployment using Docker containers.

## 📋 Project Overview

- **Model**: `distilbert-base-cased` (fine-tuned version available at `DuckyDuck123/bert-goodreads-genres`)
- **Dataset**: Goodreads Book Reviews (8-genre classification)
- **Framework**: PyTorch, Hugging Face Transformers
- **Task**: Multi-class genre classification (8 genres)

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

## 🚀 Features

- Fine-tuning DistilBERT model for multi-class text classification
- Automated training pipeline with checkpointing
- Comprehensive model evaluation with macro-averaged metrics
- Docker containerization for training and evaluation
- Model versioning and Hugging Face Hub integration
- Goodreads dataset loading and preprocessing

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

The project includes automated data loading from Goodreads:

```python
from transformers import DistilBertTokenizerFast
from src.data import load_and_prepare_data

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
train_dataset, test_dataset, label_encoder = load_and_prepare_data(tokenizer)
```

The data module handles:
- Downloading gzipped JSON reviews from Goodreads
- Sampling and splitting data into train/test sets
- Encoding genre labels
- Tokenization with DistilBERT tokenizer

## 📈 Model Performance

The fine-tuned DistilBERT model will be evaluated on the Goodreads test set with the following metrics:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro-averaged F1 score across all genres
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall

Results will be saved to `results/eval_results.json` after training completes.

## 🔧 Configuration

Key configurations in `src/train.py`:
- `MODEL_NAME`: Base model from Hugging Face (`distilbert-base-cased`)
- `OUTPUT_DIR`: Directory to save checkpoints
- `HF_REPO`: Hugging Face Hub repository name
- `MAX_LENGTH`: Maximum sequence length (512 tokens)
- `TRAIN_SIZE`: Training samples per genre (800)
- `TEST_SIZE`: Test samples per genre (200)
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

- Hugging Face Model: [DuckyDuck123/bert-goodreads-genres](https://huggingface.co/DuckyDuck123/bert-goodreads-genres)
- Base Model: [distilbert-base-cased](https://huggingface.co/distilbert-base-cased)
- Dataset: [Goodreads Book Reviews](https://mengtingwan.github.io/data/goodreads.html#datasets)

## 📚 References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [DistilBERT: Distilled version of BERT](https://arxiv.org/abs/1910.01108)
- [Goodreads Dataset Paper](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys17.pdf)
