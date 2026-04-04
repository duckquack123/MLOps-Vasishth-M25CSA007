"""
Q2(ii)(b) — Adversarial Detection Model using ResNet-34.
Input: Mix of clean + adversarial images (BIM attack via IBM ART).
Output: Binary classification — clean (0) vs adversarial (1).
Target detection accuracy ≥ 70%.

Usage:
    python detect_bim.py --checkpoint checkpoints/resnet18_cifar10_best.pt \
                         --epochs 20
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from datasets import load_dataset
from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
import wandb

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wandb_utils import login_from_env

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ass5-q2-adversarial")
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD  = np.array([0.2023, 0.1994, 0.2010])


# ─── (Shared helpers identical to detect_pgd.py) ─────────────────────────────

def load_victim_resnet18(ckpt_path: str):
    model     = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(DEVICE).eval()


def get_art_classifier(victim_model):
    return PyTorchClassifier(
        model=victim_model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(victim_model.parameters(), lr=0.01),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(MEAN, STD),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )


def load_cifar10_np(n_train: int, n_test: int):
    tf = transforms.Compose([transforms.ToTensor()])
    print(f"Loading CIFAR-10 from Hugging Face Hub (BIM Detector - n_train={n_train})")
    hf_ds = load_dataset("cifar10", token=os.environ.get("HF_TOKEN"))
    
    # Process train split
    train_slice = hf_ds['train'].select(range(min(n_train, len(hf_ds['train']))))
    X_train = np.stack([tf(item['img']).numpy() for item in train_slice])
    
    # Process test split
    test_slice = hf_ds['test'].select(range(min(n_test, len(hf_ds['test']))))
    X_test = np.stack([tf(item['img']).numpy() for item in test_slice])
    
    return X_train.astype(np.float32), X_test.astype(np.float32)


def build_detector():
    model     = models.resnet34(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def train_detector(model, train_loader, test_loader, epochs: int, lr: float,
                   ckpt_name: str = "detector_bim_best.pt"):
    mean = torch.tensor(MEAN, device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=DEVICE, dtype=torch.float32).view(1, 3, 1, 1)

    def normalize_batch(x):
        return (x - mean) / std

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_acc  = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                out  = model(normalize_batch(X_b))
                loss = criterion(out, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss    += loss.item() * X_b.size(0)
            tr_correct += (out.argmax(1) == y_b).sum().item()
            tr_total   += X_b.size(0)
        scheduler.step()

        model.eval()
        te_correct, te_total = 0, 0
        te_loss_total = 0.0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    out  = model(normalize_batch(X_b))
                    loss = criterion(out, y_b)
                te_loss_total += loss.item() * X_b.size(0)
                te_correct    += (out.argmax(1) == y_b).sum().item()
                te_total      += X_b.size(0)

        tr_acc = tr_correct / tr_total
        te_acc = te_correct / te_total
        print(f"Epoch {epoch:>3}: train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")
        wandb.log({"epoch": epoch,
                   "train/loss": tr_loss / tr_total,
                   "test/loss":  te_loss_total / te_total,
                   "train/accuracy": tr_acc,
                   "test/accuracy":  te_acc})

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), f"checkpoints/{ckpt_name}")

    return best_acc


def log_wandb_samples(X_clean, X_adv, n: int = 10, attack_name: str = "BIM"):
    imgs = []
    for i in range(min(n, len(X_clean))):
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        for ax, img, label in zip(
            axes,
            [X_clean[i], X_adv[i]],
            ["Clean (0)", f"{attack_name} Adv (1)"],
        ):
            display = (img * STD[:, None, None] + MEAN[:, None, None]).clip(0, 1)
            ax.imshow(display.transpose(1, 2, 0))
            ax.set_title(label, fontsize=7)
            ax.axis("off")
        plt.tight_layout()
        path = f"outputs/bim_sample_{i}.png"
        plt.savefig(path, dpi=100)
        plt.close()
        imgs.append(wandb.Image(path))
    wandb.log({"BIM_samples": imgs})


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    login_from_env()
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    np.random.seed(42)

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=os.environ.get("WANDB_ENTITY", None),
        name="detector_bim",
        config=vars(args),
    )

    victim_model = load_victim_resnet18(args.checkpoint)
    art_clf      = get_art_classifier(victim_model)
    X_train_raw, X_test_raw = load_cifar10_np(args.n_train, args.n_test)

    print(f"Generating BIM adversarial examples  (eps={args.eps}, steps={args.steps})…")
    bim = BasicIterativeMethod(
        estimator=art_clf,
        eps=args.eps,
        eps_step=args.eps / args.steps,
        max_iter=args.steps,
        targeted=False,
        verbose=True,
    )
    X_adv_train = bim.generate(x=X_train_raw)
    X_adv_test  = bim.generate(x=X_test_raw)

    # Mix labels
    X_tr = np.concatenate([X_train_raw, X_adv_train], axis=0)
    y_tr = np.array([0]*len(X_train_raw) + [1]*len(X_adv_train), dtype=np.int64)
    X_te = np.concatenate([X_test_raw,  X_adv_test],  axis=0)
    y_te = np.array([0]*len(X_test_raw)  + [1]*len(X_adv_test),  dtype=np.int64)

    idx_tr = np.random.permutation(len(X_tr))
    idx_te = np.random.permutation(len(X_te))
    X_tr, y_tr = X_tr[idx_tr], y_tr[idx_tr]
    X_te, y_te = X_te[idx_te], y_te[idx_te]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te), torch.tensor(y_te)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # Log 10 sample pairs
    clean_candidates = X_tr[y_tr == 0][:10]
    adv_candidates = X_tr[y_tr == 1][:10]
    if len(clean_candidates) == 10 and len(adv_candidates) == 10:
        log_wandb_samples(clean_candidates, adv_candidates, attack_name="BIM")
    else:
        half = len(X_tr) // 2
        log_wandb_samples(X_tr[:10], X_tr[half:half+10], attack_name="BIM")

    detector = build_detector()
    best_acc = train_detector(detector, train_loader, test_loader,
                              args.epochs, args.lr)

    print(f"\n✓ Best BIM detection accuracy: {best_acc*100:.2f}%")
    if best_acc < 0.70:
        print("⚠  Target ≥70% NOT reached — consider more epochs.")
    wandb.summary["best_detection_accuracy_bim"] = best_acc
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Detector — BIM attack")
    parser.add_argument("--checkpoint",  default="checkpoints/resnet18_cifar10_best.pt")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--eps",         type=float, default=0.03)
    parser.add_argument("--steps",       type=int,   default=10)
    parser.add_argument("--n_train",     type=int,   default=10000)
    parser.add_argument("--n_test",      type=int,   default=2000)
    args = parser.parse_args()
    main(args)
