import os
import torch

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "WinKawaks/vit-small-patch16-224"
NUM_CLASSES = 100
IMAGE_SIZE  = 224

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_EPOCHS   = 10
BASE_LR      = 1e-3
WEIGHT_DECAY = 1e-4

# ─── LoRA hyper-params ────────────────────────────────────────────────────────
LORA_RANKS          = [2, 4, 8]
LORA_ALPHAS         = [2, 4, 8]
LORA_DROPOUT        = 0.1
LORA_TARGET_MODULES = ["query", "key", "value"]

# ─── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── WandB ────────────────────────────────────────────────────────────────────
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ass5-q1-vit-lora")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  None)

# ─── HuggingFace ──────────────────────────────────────────────────────────────
HF_USERNAME = os.environ.get("HF_USERNAME", "DuckyDuck123")

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
