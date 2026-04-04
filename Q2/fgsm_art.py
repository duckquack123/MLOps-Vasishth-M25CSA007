"""
Q2(i) — FGSM attack using IBM Adversarial Robustness Toolbox (ART).
Compares clean accuracy vs adversarial accuracy (with and without ART).
Also uploads 10 WandB sample images for each category.

Usage:
    python fgsm_art.py --checkpoint checkpoints/resnet18_cifar10_best.pt
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
from torchvision import datasets, models, transforms

# IBM ART imports
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import wandb

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wandb_utils import login_from_env

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ass5-q2-adversarial")
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]
MEAN = np.array([0.4914, 0.4822, 0.4465])
STD  = np.array([0.2023, 0.1994, 0.2010])


# ─── Data ─────────────────────────────────────────────────────────────────────

def get_raw_test_data(n: int = 1000):
    """Return (X_norm, y) numpy arrays: X_norm in [0,1] after normalisation."""
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=n, shuffle=False)
    X, y = next(iter(loader))
    # Normalise to match training distribution — ART works on raw [0,1] range
    # but the classifier wrapper takes normalised images.
    # We'll normalise here and pass (mean, std) to ART via preprocessing_defences=None
    X_norm = ((X.numpy() - MEAN[:, None, None]) / STD[:, None, None])
    return X_norm.astype(np.float32), y.numpy()


# ─── Model ────────────────────────────────────────────────────────────────────

def load_resnet18(ckpt_path: str):
    model     = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(DEVICE).eval()


# ─── ART Classifier Wrapper ───────────────────────────────────────────────────

def build_art_classifier(model):
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.SGD(model.parameters(), lr=0.01)

    # ART expects input in the same space the model was trained on.
    # Here that's already-normalised images, so clip_values cover the normalised range.
    clip_lo = ((0 - MEAN) / STD).min()
    clip_hi = ((1 - MEAN) / STD).max()

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(float(clip_lo), float(clip_hi)),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )
    return classifier


# ─── Helpers ──────────────────────────────────────────────────────────────────

def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return (predictions.argmax(1) == labels).mean()


def denorm(x: np.ndarray) -> np.ndarray:
    """x is (C,H,W) normalised → uint8 (H,W,C)."""
    img = (x * STD[:, None, None] + MEAN[:, None, None]).clip(0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def save_comparison_grid(X_clean, X_adv, y, preds_clean, preds_adv,
                         title: str, n: int = 10, save_path: str = "grid.png"):
    fig, axes = plt.subplots(2, n, figsize=(2*n, 5))
    for i in range(n):
        axes[0, i].imshow(denorm(X_clean[i]))
        axes[0, i].set_title(f"Clean\n{CIFAR10_CLASSES[preds_clean[i]]}", fontsize=7)
        axes[0, i].axis("off")
        axes[1, i].imshow(denorm(X_adv[i]))
        axes[1, i].set_title(f"Adv\n{CIFAR10_CLASSES[preds_adv[i]]}", fontsize=7)
        axes[1, i].axis("off")
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    login_from_env()
    os.makedirs("outputs", exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=os.environ.get("WANDB_ENTITY", None),
        name="fgsm_art",
        config=vars(args),
    )

    model      = load_resnet18(args.checkpoint)
    classifier = build_art_classifier(model)
    X, y       = get_raw_test_data(n=1000)

    # ── Clean accuracy ────────────────────────────────────────────────────────
    clean_preds = classifier.predict(X, batch_size=128)
    clean_acc   = accuracy(clean_preds, y)
    print(f"Clean accuracy: {clean_acc*100:.2f}%")
    wandb.log({"clean_accuracy": clean_acc})

    table_rows = []
    epsilons   = [0.01, 0.02, 0.03, 0.05, 0.1]

    for eps in epsilons:
        fgsm    = FastGradientMethod(estimator=classifier, eps=eps)
        X_adv   = fgsm.generate(x=X)
        adv_preds = classifier.predict(X_adv, batch_size=128)
        adv_acc   = accuracy(adv_preds, y)
        drop      = (clean_acc - adv_acc) * 100

        print(f"  ε={eps:.2f}: adv_acc={adv_acc*100:.2f}%  drop={drop:.2f}%")
        wandb.log({"epsilon": eps, "adv_accuracy_art": adv_acc,
                   "accuracy_drop_art": drop})
        table_rows.append((eps, adv_acc, drop))

    # ── Log 10 WandB sample images for ε=0.03 ─────────────────────────────────
    vis_eps    = 0.03
    fgsm_vis   = FastGradientMethod(estimator=classifier, eps=vis_eps)
    X_adv_vis  = fgsm_vis.generate(x=X[:10])
    preds_c    = clean_preds[:10].argmax(1)
    preds_a    = classifier.predict(X_adv_vis).argmax(1)

    wandb_imgs = []
    for i in range(10):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
        ax1.imshow(denorm(X[i]))
        ax1.set_title(f"Clean: {CIFAR10_CLASSES[preds_c[i]]}", fontsize=8)
        ax1.axis("off")
        ax2.imshow(denorm(X_adv_vis[i]))
        ax2.set_title(f"FGSM ART: {CIFAR10_CLASSES[preds_a[i]]}", fontsize=8)
        ax2.axis("off")
        plt.tight_layout()
        path = f"outputs/fgsm_art_sample_{i}.png"
        plt.savefig(path, dpi=100)
        plt.close()
        wandb_imgs.append(wandb.Image(path,
            caption=f"True:{CIFAR10_CLASSES[y[i]]} | Clean:{CIFAR10_CLASSES[preds_c[i]]} | Adv:{CIFAR10_CLASSES[preds_a[i]]}"))

    wandb.log({"fgsm_art_samples": wandb_imgs})

    # ── Save visual grid ──────────────────────────────────────────────────────
    grid_path = f"outputs/fgsm_art_eps{vis_eps}.png"
    save_comparison_grid(X[:10], X_adv_vis, y[:10], preds_c, preds_a,
                         title=f"FGSM via IBM ART  |  ε = {vis_eps}",
                         save_path=grid_path)
    wandb.log({"fgsm_art_grid": wandb.Image(grid_path)})

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'ε':>6} {'Adv Acc (%)':>12} {'Drop (%)':>10}")
    print("-" * 32)
    print(f"{'0.00':>6} {clean_acc*100:>12.2f} {'—':>10}")
    for eps, adv_acc, drop in table_rows:
        print(f"{eps:>6.2f} {adv_acc*100:>12.2f} {drop:>10.2f}")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FGSM Attack via IBM ART on CIFAR-10")
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_cifar10_best.pt")
    args = parser.parse_args()
    main(args)
