import os
from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
ALLOW_PICKLE = False

# Project root (keep)
root = Path(__file__).parent.parent


def _resolve_path(env_key: str, default_rel: str) -> Path:
    """Return path from environment or default (relative to project root if not absolute)."""
    val = os.environ.get(env_key, default_rel)
    p = Path(val)
    if not p.is_absolute():
        p = root / p
    return p


# Defaults with environment override
DATA_PATH = _resolve_path("DATA_PATH", "data")
TRAINING_PATH = _resolve_path("TRAINING_PATH", "outputs/training")
EVAL_PATH = _resolve_path("EVAL_PATH", "outputs/results")
WANDB_PATH = _resolve_path("WANDB_PATH", "wandb")
THIRD_PARTY_PATH = _resolve_path("THIRD_PARTY_PATH", "third_party")
