"""
Q2 — Train ResNet-18 from scratch on CIFAR-10.
Target: ≥ 72% test classification accuracy.

Usage:
    python train_resnet18.py [--epochs 30] [--batch_size 128]
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
import wandb

os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("WANDB_CONSOLE", "off")
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wandb_utils import login_from_env

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints/resnet18_cifar10_best.pt"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ass5-q2-adversarial")

class HFToTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['img']
        label = item['label']
        if self.transform:
            img = self.transform(img)
        return img, label

# ─── Data ─────────────────────────────────────────────────────────────────────

def get_loaders(batch_size: int):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    print("Loading CIFAR-10 from Hugging Face Hub...")
    hf_ds = load_dataset("cifar10", token=os.environ.get("HF_TOKEN"))
    
    train_ds = HFToTorchDataset(hf_ds['train'], transform=train_tf)
    test_ds  = HFToTorchDataset(hf_ds['test'], transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, test_loader


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model():
    """ResNet-18 adapted for CIFAR-10 (32×32): replace stem conv + remove max-pool."""
    model = models.resnet18(weights=None)
    # CIFAR adaptation: smaller initial conv, no max-pool
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)


# ─── Train / Eval ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out  = model(images)
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
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast('cuda'):
            out  = model(images)
            loss = criterion(out, labels)
        total_loss += loss.item() * images.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    login_from_env()
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=os.environ.get("WANDB_ENTITY", None),
        name="resnet18_cifar10_scratch",
        config=vars(args),
    )

    os.makedirs("checkpoints", exist_ok=True)
    train_loader, test_loader = get_loaders(args.batch_size)
    model     = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=len(train_loader),
    )
    scaler    = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    print(f"\n{'Epoch':>5} {'TrLoss':>8} {'TeLoss':>8} {'TrAcc':>7} {'TeAcc':>7}")
    print("-" * 42)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        print(f"{epoch:>5} {tr_loss:>8.4f} {te_loss:>8.4f} {tr_acc:>7.4f} {te_acc:>7.4f}")
        wandb.log({
            "epoch":          epoch,
            "train/loss":     tr_loss,
            "test/loss":      te_loss,
            "train/accuracy": tr_acc,
            "test/accuracy":  te_acc,
        })

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ New best: {best_acc:.4f}")

    print(f"\nBest test accuracy: {best_acc:.4f}")
    if best_acc < 0.72:
        print("⚠  Target ≥72% NOT reached — consider more epochs or LR tuning.")
    wandb.summary["best_test_accuracy"] = best_acc
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet-18 from scratch on CIFAR-10")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=0.1)
    args = parser.parse_args()
    main(args)
