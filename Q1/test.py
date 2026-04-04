"""
Q1 — Testing / Evaluation script.
Evaluates any checkpoint on CIFAR-100 test set, prints per-class accuracy,
generates a histogram, and logs a summary table to WandB.

Usage:
    # Evaluate baseline:
    python test.py --checkpoint checkpoints/baseline_best.pt --lora False

    # Evaluate LoRA checkpoint (PEFT saved directory):
    python test.py --checkpoint checkpoints/lora_r4_a4_d0.1_best \
                   --lora True --rank 4 --alpha 4 --dropout 0.1
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import ViTForImageClassification
from peft import PeftModel
import wandb

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wandb_utils import login_from_env

class HFToTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['img']
        label = item['fine_label']
        if self.transform:
            img = self.transform(img)
        return img, label

CIFAR100_CLASSES = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle",
    "bottle","bowl","boy","bridge","bus","butterfly","camel","can","castle",
    "caterpillar","cattle","chair","chimpanzee","clock","cloud","cockroach","couch",
    "crab","crocodile","cup","dinosaur","dolphin","elephant","flatfish","forest",
    "fox","girl","hamster","house","kangaroo","keyboard","lamp","lawn_mower",
    "leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear",
    "pickup_truck","pine_tree","plain","plate","poppy","porcupine","possum",
    "rabbit","raccoon","ray","road","rocket","rose","sea","seal","shark","shrew",
    "skunk","skyscraper","snail","snake","spider","squirrel","streetcar",
    "sunflower","sweet_pepper","table","tank","telephone","television","tiger",
    "tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree",
    "wolf","woman","worm",
]


def get_test_loader(batch_size: int):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    tf   = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    print("Loading CIFAR-100 Test from Hugging Face Hub...")
    hf_ds = load_dataset("cifar100", split="test", token=os.environ.get("HF_TOKEN"))
    ds = HFToTorchDataset(hf_ds, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True)


def load_model(args):
    if args.lora:
        base = ViTForImageClassification.from_pretrained(
            cfg.MODEL_NAME, num_labels=cfg.NUM_CLASSES, ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base, args.checkpoint)
    else:
        model = ViTForImageClassification.from_pretrained(
            cfg.MODEL_NAME, num_labels=cfg.NUM_CLASSES, ignore_mismatched_sizes=True,
        )
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
    return model.to(cfg.DEVICE)


@torch.no_grad()
def evaluate_classwise(model, loader):
    model.eval()
    class_correct = np.zeros(cfg.NUM_CLASSES)
    class_total   = np.zeros(cfg.NUM_CLASSES)

    for images, labels in loader:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        with torch.amp.autocast('cuda'):
            preds = model(pixel_values=images).logits.argmax(1)
        for c in range(cfg.NUM_CLASSES):
            mask = labels == c
            class_correct[c] += (preds[mask] == c).sum().item()
            class_total[c]   += mask.sum().item()

    per_class = class_correct / np.maximum(class_total, 1)
    overall   = class_correct.sum() / class_total.sum()
    return per_class, overall


def plot_classwise_histogram(per_class, save_path: str = "classwise_accuracy.png"):
    fig, ax = plt.subplots(figsize=(24, 5))
    x = np.arange(cfg.NUM_CLASSES)
    bars = ax.bar(x, per_class * 100, color=plt.cm.viridis(per_class))
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR100_CLASSES, rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Class-wise Test Accuracy on CIFAR-100")
    ax.set_ylim(0, 110)
    ax.axhline(per_class.mean() * 100, color="red", linestyle="--",
               label=f"Mean = {per_class.mean()*100:.1f}%")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def main(args):
    login_from_env()
    run = wandb.init(
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        name=f"test_{os.path.basename(args.checkpoint)}",
        config=vars(args),
    )

    loader      = get_test_loader(args.batch_size)
    model       = load_model(args)
    n_train, n_total = count_trainable(model)

    per_class, overall = evaluate_classwise(model, loader)
    print(f"\nOverall Test Accuracy : {overall*100:.2f}%")
    print(f"Trainable Params      : {n_train:,}  /  Total: {n_total:,}")

    # ── Histogram ─────────────────────────────────────────────────────────────
    hist_path = plot_classwise_histogram(per_class)
    wandb.log({"classwise_accuracy_histogram": wandb.Image(hist_path)})
    print(f"Saved histogram → {hist_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    has_lora = "yes" if args.lora else "no"
    rank     = args.rank    if args.lora else "—"
    alpha    = args.alpha   if args.lora else "—"
    dropout  = args.dropout if args.lora else "—"

    row = {
        "LoRA layers":         has_lora,
        "Rank":                rank,
        "Alpha":               alpha,
        "Dropout":             dropout,
        "Overall Test Acc (%)":f"{overall*100:.2f}",
        "Trainable Params":    f"{n_train:,}",
    }
    print("\n" + "─"*60)
    for k, v in row.items():
        print(f"  {k:<25}: {v}")
    print("─"*60)

    wandb.summary.update({
        "test_accuracy":    overall,
        "trainable_params": n_train,
        **row,
    })

    # Save row to JSON for README tables
    result_file = os.path.join(cfg.CHECKPOINT_DIR,
                               f"test_result_{os.path.basename(args.checkpoint)}.json")
    with open(result_file, "w") as f:
        json.dump(row, f, indent=2)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViT-S on CIFAR-100 test set")
    parser.add_argument("--checkpoint",  required=True, help="Path to checkpoint")
    parser.add_argument("--lora",        type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--rank",        type=int,   default=4)
    parser.add_argument("--alpha",       type=float, default=4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--batch_size",  type=int,   default=cfg.BATCH_SIZE)
    args = parser.parse_args()
    main(args)
