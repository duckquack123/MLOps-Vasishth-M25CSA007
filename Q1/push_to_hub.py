"""
Q1 — Push best LoRA model weights to HuggingFace Hub.

Usage:
    python push_to_hub.py --checkpoint checkpoints/lora_r4_a4_d0.1_best \
                          --repo_name vit-small-cifar100-lora
"""

import argparse
import os
import sys

from huggingface_hub import HfApi
from peft import PeftModel
from transformers import ViTForImageClassification

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main(args):
    HF_USERNAME = os.environ.get("HF_USERNAME", cfg.HF_USERNAME)
    HF_TOKEN    = os.environ.get("HF_TOKEN", None)

    if not HF_TOKEN:
        print("⚠  HF_TOKEN env var not set. Attempting login via huggingface-cli...")

    repo_id = f"{HF_USERNAME}/{args.repo_name}"
    print(f"Pushing to HuggingFace: {repo_id}")

    # Load base + LoRA
    base  = ViTForImageClassification.from_pretrained(
        cfg.MODEL_NAME, num_labels=cfg.NUM_CLASSES, ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base, args.checkpoint)

    # Push
    model.push_to_hub(repo_id, token=HF_TOKEN)
    print(f"✓ Model pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push LoRA model to HuggingFace Hub")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to PEFT saved directory")
    parser.add_argument("--repo_name",  default="vit-small-cifar100-lora",
                        help="HuggingFace repo name (under your username)")
    args = parser.parse_args()
    main(args)
