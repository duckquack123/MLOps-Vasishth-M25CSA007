import os
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer
)
import evaluate

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "DuckyDuck123/bert-tiny-imdb"
RESULTS_DIR = "../results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("imdb")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

test_dataset = tokenized_dataset["test"]

# -----------------------------
# Load model from Hugging Face
# -----------------------------
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# -----------------------------
# Metrics
# -----------------------------
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# -----------------------------
# Trainer for evaluation only
# -----------------------------
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Evaluate
# -----------------------------
eval_results = trainer.evaluate(eval_dataset=test_dataset)

print("\nEvaluation Results:", eval_results)

# Save results
results_path = os.path.join(RESULTS_DIR, "hf_eval_results.json")

with open(results_path, "w") as f:
    json.dump(eval_results, f, indent=4)

print(f"\nResults saved to {results_path}")
