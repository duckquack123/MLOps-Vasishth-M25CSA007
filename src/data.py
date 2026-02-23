from datasets import load_dataset
from transformers import BertTokenizer

MODEL_NAME = "prajjwal1/bert-tiny"

def load_and_prepare_data():

    dataset = load_dataset("imdb")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    dataset = dataset.map(tokenize, batched=True)

    dataset = dataset.rename_column("label", "labels")

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    train_dataset = dataset["train"].select(range(2000))
    test_dataset = dataset["test"].select(range(1000))

    return train_dataset, test_dataset, tokenizer

