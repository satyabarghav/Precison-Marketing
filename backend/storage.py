from __future__ import annotations

from pathlib import Path
from typing import Optional

from joblib import dump, load


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_model(path: str, model) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = MODELS_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    dump(model, p)
    return str(p)


def load_model(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = MODELS_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {p}")
    return load(p)


