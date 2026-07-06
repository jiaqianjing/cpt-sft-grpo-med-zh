"""Config + secrets loading (I/O layer, kept out of schema.py per LOD Pattern 6).

CALLING SPEC:
    from configs.loader import load_config, load_secrets
    cfg = load_config()                        # ProjectConfig defaults
    cfg = load_config("configs/experiment.yaml")  # overlay a YAML onto defaults
    secrets = load_secrets()                   # -> {"WANDB_API_KEY": bool, ...} presence map
                                               # (also exports the vars into os.environ)

Side effects: load_secrets() reads .env and sets os.environ for any keys found.
Deterministic given the same files. No network.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from configs.schema import ProjectConfig

_SECRET_KEYS = ("WANDB_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN")


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load ProjectConfig; overlay a YAML file if given. Raises ValidationError on bad values."""
    if path is None:
        return ProjectConfig()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return ProjectConfig(**data)


def load_secrets(env_path: str | Path = ".env") -> dict[str, bool]:
    """Parse a simple KEY=VALUE .env, export into os.environ, return a presence map.

    Does not overwrite variables already set in the real environment.
    """
    p = Path(env_path)
    present: dict[str, bool] = {k: bool(os.environ.get(k)) for k in _SECRET_KEYS}
    if not p.exists():
        return present
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = val
        if key in _SECRET_KEYS:
            present[key] = bool(os.environ.get(key))
    return present
