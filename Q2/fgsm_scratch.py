"""
Q2(i) — FGSM Attack from Scratch (no ART).
Evaluates a trained ResNet-18 on clean and adversarial CIFAR-10 images,
sweeping epsilon values. Saves visual comparison grids.

Usage:
    python fgsm_scratch.py --checkpoint checkpoints/resnet18_cifar10_best.pt
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import wandb

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wandb_utils import login_from_env

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ass5-q2-adversarial")
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]


# ─── Data ─────────────────────────────────────────────────────────────────────

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def get_test_loader(batch_size: int = 128):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN.squeeze().tolist(), STD.squeeze().tolist()),
    ])
    ds = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


# ─── Model ────────────────────────────────────────────────────────────────────

def load_resnet18(ckpt_path: str):
    model     = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(DEVICE).eval()


# ─── FGSM from Scratch ────────────────────────────────────────────────────────

def fgsm_attack(model, images, labels, epsilon: float) -> torch.Tensor:
    """Untargeted FGSM: x_adv = x + ε · sign(∇_x L(x, y))"""
    images  = images.clone().requires_grad_(True)
    loss    = nn.CrossEntropyLoss()(model(images), labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    adv_images   = images.detach() + perturbation
    # Clamp to valid normalised range
    lower = ((0 - MEAN.to(DEVICE)) / STD.to(DEVICE))
    upper = ((1 - MEAN.to(DEVICE)) / STD.to(DEVICE))
    adv_images = torch.max(torch.min(adv_images, upper), lower)
    return adv_images.detach()


# ─── Evaluation ───────────────────────────────────────────────────────────────

def accuracy_on_loader(model, loader) -> float:
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def accuracy_under_fgsm(model, loader, epsilon: float) -> float:
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        adv            = fgsm_attack(model, images, labels, epsilon)
        correct       += (model(adv).argmax(1) == labels).sum().item()
        total         += labels.size(0)
    return correct / total


# ─── Visualisation ────────────────────────────────────────────────────────────

def denorm(t: torch.Tensor) -> np.ndarray:
    """Denorm a (C,H,W) tensor to numpy (H,W,C) uint8."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std  = torch.tensor([0.2023, 0.1994, 0.2010])
    img  = t.cpu() * std[:, None, None] + mean[:, None, None]
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def save_visual_grid(model, loader, epsilon: float, n: int = 10,
                     save_path: str = "fgsm_scratch_grid.png"):
    images, labels = next(iter(loader))
    images, labels = images[:n].to(DEVICE), labels[:n].to(DEVICE)
    advs = fgsm_attack(model, images, labels, epsilon)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    with torch.no_grad():
        clean_preds = model(images).argmax(1).cpu()
        adv_preds   = model(advs).argmax(1).cpu()

    for i in range(n):
        axes[0, i].imshow(denorm(images[i]))
        axes[0, i].set_title(f"Clean\n{CIFAR10_CLASSES[clean_preds[i]]}", fontsize=7)
        axes[0, i].axis("off")
        axes[1, i].imshow(denorm(advs[i]))
        axes[1, i].set_title(f"Adv(ε={epsilon})\n{CIFAR10_CLASSES[adv_preds[i]]}", fontsize=7)
        axes[1, i].axis("off")

    plt.suptitle(f"FGSM from Scratch  |  ε = {epsilon}", fontsize=12)
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
        name="fgsm_scratch",
        config=vars(args),
    )

    model  = load_resnet18(args.checkpoint)
    loader = get_test_loader()

    clean_acc = accuracy_on_loader(model, loader)
    print(f"Clean accuracy: {clean_acc*100:.2f}%")
    wandb.log({"clean_accuracy": clean_acc, "epsilon": 0.0})

    epsilons = [0.01, 0.02, 0.03, 0.05, 0.1]
    rows     = []

    for eps in epsilons:
        adv_acc = accuracy_under_fgsm(model, loader, eps)
        drop    = (clean_acc - adv_acc) * 100
        print(f"  ε={eps:.2f}: adv_acc={adv_acc*100:.2f}%  drop={drop:.2f}%")
        wandb.log({"epsilon": eps, "adv_accuracy_scratch": adv_acc,
                   "accuracy_drop_scratch": drop})
        rows.append({"epsilon": eps, "adv_acc": adv_acc, "drop": drop})

    # ── Visual comparison for fixed ε ─────────────────────────────────────────
    grid_eps  = epsilons[2]   # ε = 0.03 for visual
    grid_path = f"outputs/fgsm_scratch_eps{grid_eps}.png"
    save_visual_grid(model, loader, grid_eps, save_path=grid_path)
    wandb.log({"fgsm_scratch_grid": wandb.Image(grid_path)})

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'ε':>6} {'Adv Acc (%)':>12} {'Drop (%)':>10}")
    print("-" * 32)
    print(f"{'0.00':>6} {clean_acc*100:>12.2f} {'—':>10}")
    for r in rows:
        print(f"{r['epsilon']:>6.2f} {r['adv_acc']*100:>12.2f} {r['drop']:>10.2f}")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FGSM Attack from Scratch on CIFAR-10")
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_cifar10_best.pt")
    args = parser.parse_args()
    main(args)
