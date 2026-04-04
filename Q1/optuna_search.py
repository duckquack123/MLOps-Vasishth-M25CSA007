"""
Q1 — Optuna hyperparameter search for LoRA on ViT-S / CIFAR-100.
Searches LoRA hyperparameters only, keeping the assignment dropout fixed.

Usage:
    python optuna_search.py [--n_trials 20] [--epochs 5]
"""

import argparse
import os
import sys

import optuna
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
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

# keep loaders module-level so they're built once
_train_loader = None
_val_loader   = None


def _get_loaders(batch_size: int):
    global _train_loader, _val_loader
    if _train_loader is None:
        mean = (0.5071, 0.4867, 0.4408)
        std  = (0.2675, 0.2565, 0.2761)

        train_tf = transforms.Compose([
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        print("Loading CIFAR-100 from Hugging Face Hub (Optuna)...")
        hf_ds = load_dataset("cifar100", token=os.environ.get("HF_TOKEN"))
        
        train_ds = HFToTorchDataset(hf_ds['train'], transform=train_tf)
        val_ds   = HFToTorchDataset(hf_ds['test'], transform=val_tf)

        _train_loader = DataLoader(
            train_ds,
            batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
        )
        _val_loader = DataLoader(
            val_ds,
            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
        )
    return _train_loader, _val_loader


def _train_eval(rank, alpha, dropout, epochs, batch_size, lr):
    """Train for `epochs` and return best val accuracy."""
    base = ViTForImageClassification.from_pretrained(
        cfg.MODEL_NAME, num_labels=cfg.NUM_CLASSES, ignore_mismatched_sizes=True,
    )
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=cfg.LORA_TARGET_MODULES,
        lora_dropout=dropout, bias="none",
        modules_to_save=["classifier"],
    )
    model     = get_peft_model(base, lora_cfg).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler('cuda')
    train_loader, val_loader = _get_loaders(batch_size)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                loss = criterion(model(pixel_values=images).logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # val accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                with torch.amp.autocast('cuda'):
                    preds = model(pixel_values=images).logits.argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        best_acc = max(best_acc, acc)

    del model
    torch.cuda.empty_cache()
    return best_acc


def objective(trial, epochs, batch_size):
    rank    = trial.suggest_categorical("rank", [2, 4, 8])
    alpha   = trial.suggest_categorical("alpha", [2, 4, 8])
    dropout = cfg.LORA_DROPOUT
    lr      = cfg.BASE_LR

    print(f"\n[Trial {trial.number}] rank={rank}, alpha={alpha}, "
          f"dropout={dropout}, lr={lr:.2e}")

    best_acc = _train_eval(rank, alpha, dropout, epochs, batch_size, lr)

    # log to WandB
    wandb.log({
        "trial":      trial.number,
        "rank":       rank,
        "alpha":      alpha,
        "dropout":    dropout,
        "lr":         lr,
        "val_acc":    best_acc,
    })
    return best_acc


def main(args):
    login_from_env()
    wandb.init(
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        name="optuna_search",
        config={"n_trials": args.n_trials, "epochs_per_trial": args.epochs},
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="vit_lora_cifar100",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: objective(t, args.epochs, args.batch_size),
        n_trials=args.n_trials,
    )

    best = study.best_trial
    print("\n" + "="*50)
    print("Best trial:")
    print(f"  Val Accuracy : {best.value:.4f}")
    print(f"  Params       : {best.params}")

    wandb.summary["best_val_accuracy"] = best.value
    wandb.summary["best_params"]       = best.params

    # save best config
    import json
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(cfg.CHECKPOINT_DIR, "optuna_best_params.json"), "w") as f:
        json.dump({"val_accuracy": best.value, "params": best.params}, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna LoRA Hyperparameter Search")
    parser.add_argument("--n_trials",   type=int, default=20,
                        help="Number of Optuna trials")
    parser.add_argument("--epochs",     type=int, default=5,
                        help="Epochs per trial (shorter than full training)")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    args = parser.parse_args()
    main(args)
