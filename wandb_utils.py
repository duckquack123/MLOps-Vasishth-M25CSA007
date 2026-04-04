import os

import wandb


def login_from_env() -> None:
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True, verify=True)