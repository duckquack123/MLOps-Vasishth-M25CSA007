import os
import numpy as np
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import evaluate

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "prajjwal1/bert-tiny"
OUTPUT_DIR = "models"
RESULTS_DIR = "results"
HF_REPO = "DuckyDuck123/bert-tiny-imdb"

os.makedirs(OUTPUT_DIR, exist_ok=True)
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

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

# -----------------------------
# Load model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

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
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=100,
    push_to_hub=False
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Evaluate
# -----------------------------
eval_results = trainer.evaluate()

print("\nEvaluation Results:", eval_results)

# Save evaluation results
results_path = os.path.join(RESULTS_DIR, "eval_results.json")

import json
with open(results_path, "w") as f:
    json.dump(eval_results, f, indent=4)

# -----------------------------
# Save model
# -----------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to {OUTPUT_DIR}")
print(f"Results saved to {results_path}")
