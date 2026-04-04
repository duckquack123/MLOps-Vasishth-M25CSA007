"""
Q1 — Baseline fine-tuning: only the classification head is updated.
No LoRA involved.

Usage:
    python train.py [--epochs 10] [--batch_size 32] [--lr 1e-3]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import ViTForImageClassification
import wandb

os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB_CONSOLE", "off")
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wandb_utils import login_from_env


# ─── Data ─────────────────────────────────────────────────────────────────────

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

def get_loaders(batch_size: int):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(cfg.IMAGE_SIZE, padding=16),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    print("Loading CIFAR-100 from Hugging Face Hub...")
    hf_ds = load_dataset("cifar100", token=os.environ.get("HF_TOKEN"))
    
    train_ds = HFToTorchDataset(hf_ds['train'], transform=train_tf)
    val_ds   = HFToTorchDataset(hf_ds['test'], transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model():
    """ViT-S pre-trained on ImageNet; only the classification head is trainable."""
    model = ViTForImageClassification.from_pretrained(
        cfg.MODEL_NAME,
        num_labels=cfg.NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    # Freeze everything except the classifier head
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Baseline] Trainable params: {trainable:,} / {total:,}")
    return model.to(cfg.DEVICE)


# ─── Train / Eval ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            out  = model(pixel_values=images).logits
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        with torch.amp.autocast('cuda'):
            out  = model(pixel_values=images).logits
            loss = criterion(out, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    login_from_env()
    run = wandb.init(
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        name="baseline_no_lora",
        config={
            "model":      cfg.MODEL_NAME,
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "lr":         args.lr,
            "lora":       False,
        },
    )

    train_loader, val_loader = get_loaders(args.batch_size)
    model     = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    save_path    = os.path.join(cfg.CHECKPOINT_DIR, "baseline_best.pt")

    print(f"\n{'Epoch':>5} {'TrLoss':>8} {'VaLoss':>8} {'TrAcc':>7} {'VaAcc':>7}")
    print("-" * 42)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"{epoch:>5} {tr_loss:>8.4f} {va_loss:>8.4f} {tr_acc:>7.4f} {va_acc:>7.4f}")
        wandb.log({
            "epoch":          epoch,
            "train/loss":     tr_loss,
            "val/loss":       va_loss,
            "train/accuracy": tr_acc,
            "val/accuracy":   va_acc,
            "lr":             scheduler.get_last_lr()[0],
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best val acc: {best_val_acc:.4f}  → saved to {save_path}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    wandb.summary["best_val_accuracy"] = best_val_acc
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT-S Baseline Fine-tuning on CIFAR-100")
    parser.add_argument("--epochs",     type=int,   default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=cfg.BASE_LR)
    args = parser.parse_args()
    main(args)
