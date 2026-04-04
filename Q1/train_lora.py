"""
Q1 — LoRA fine-tuning of ViT-S on CIFAR-100.
Injects LoRA into Q, K, V attention weights via PEFT.
Runs all rank × alpha combinations sequentially (dropout fixed at 0.1).

Usage:
    # Run a specific combination:
    python train_lora.py --rank 4 --alpha 4 --dropout 0.1

    # Run ALL 9 combinations automatically:
    python train_lora.py --run_all
"""

import argparse
import itertools
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
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

def build_lora_model(rank: int, alpha: float, dropout: float):
    """ViT-S with LoRA injected into Q, K, V + trainable classification head."""
    base = ViTForImageClassification.from_pretrained(
        cfg.MODEL_NAME,
        num_labels=cfg.NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=cfg.LORA_TARGET_MODULES,   # ["query", "key", "value"]
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["classifier"],            # keep classifier fully trainable
    )
    model = get_peft_model(base, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[LoRA r={rank} α={alpha}] Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")
    return model.to(cfg.DEVICE), trainable, total


# ─── Train / Eval ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch, run):
    """Returns (loss, accuracy) and logs LoRA weight gradient histograms to WandB."""
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

    # ── Log LoRA weight gradient histograms ───────────────────────────────────
    grad_histograms = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.grad is not None:
            grad_histograms[f"grad/{name}"] = wandb.Histogram(
                param.grad.detach().cpu().float().numpy()
            )
    if grad_histograms:
        wandb.log({"epoch": epoch, **grad_histograms})

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


# ─── Single Run ───────────────────────────────────────────────────────────────

def run_experiment(rank: int, alpha: float, dropout: float,
                   epochs: int, batch_size: int, lr: float, exp_no: int = 0):
    run_name = f"lora_r{rank}_a{alpha}_d{dropout}"

    run = wandb.init(
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        name=run_name,
        config={
            "model":       cfg.MODEL_NAME,
            "lora":        True,
            "rank":        rank,
            "alpha":       alpha,
            "dropout":     dropout,
            "epochs":      epochs,
            "batch_size":  batch_size,
            "lr":          lr,
            "experiment":  exp_no,
        },
    )

    train_loader, val_loader = get_loaders(batch_size)
    model, trainable_params, total_params = build_lora_model(rank, alpha, dropout)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    save_path    = os.path.join(cfg.CHECKPOINT_DIR, f"{run_name}_best.pt")
    results      = []

    print(f"\n{'Epoch':>5} {'TrLoss':>8} {'VaLoss':>8} {'TrAcc':>7} {'VaAcc':>7}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler,
                                          criterion, epoch, run)
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
        results.append((epoch, tr_loss, va_loss, tr_acc, va_acc))

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            model.save_pretrained(save_path.replace(".pt", ""))
            print(f"  ✓ New best: {best_val_acc:.4f}")

    wandb.summary["best_val_accuracy"]  = best_val_acc
    wandb.summary["trainable_params"]   = trainable_params
    wandb.summary["total_params"]       = total_params
    run.finish()

    return best_val_acc, trainable_params, results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    login_from_env()
    if args.run_all:
        combos  = list(itertools.product(cfg.LORA_RANKS, cfg.LORA_ALPHAS))
        summary = []
        for i, (r, a) in enumerate(combos, start=1):
            print(f"\n{'='*60}")
            print(f"  Experiment {i}/{len(combos)}: rank={r}, alpha={a}, dropout={args.dropout}")
            print(f"{'='*60}")
            best_acc, n_trainable, _ = run_experiment(
                rank=r, alpha=a, dropout=args.dropout,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, exp_no=i,
            )
            summary.append({
                "exp":              i,
                "rank":             r,
                "alpha":            a,
                "dropout":          args.dropout,
                "best_val_acc":     best_acc,
                "trainable_params": n_trainable,
            })

        print("\n\n" + "="*70)
        print(f"{'Exp':>4} {'Rank':>5} {'Alpha':>6} {'Dropout':>8} "
              f"{'BestValAcc':>11} {'TrainableParams':>16}")
        print("-" * 70)
        for s in summary:
            print(f"{s['exp']:>4} {s['rank']:>5} {s['alpha']:>6} {s['dropout']:>8.2f} "
                  f"{s['best_val_acc']:>11.4f} {s['trainable_params']:>16,}")
    else:
        run_experiment(
            rank=args.rank, alpha=args.alpha, dropout=args.dropout,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT-S LoRA Fine-tuning on CIFAR-100")
    parser.add_argument("--run_all",    action="store_true", help="Run all 9 combinations")
    parser.add_argument("--rank",       type=int,   default=4)
    parser.add_argument("--alpha",      type=float, default=4)
    parser.add_argument("--dropout",    type=float, default=cfg.LORA_DROPOUT)
    parser.add_argument("--epochs",     type=int,   default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=cfg.BASE_LR)
    args = parser.parse_args()
    main(args)
