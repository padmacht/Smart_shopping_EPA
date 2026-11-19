"""
Central configuration: project paths and the dataset map from configs/config.yaml.

Usage:
    from shopsmart.config import CONFIG, PROJECT_ROOT, path
    path("raw_data")            # -> C:/Smart_shopping_EPA/data/raw
    dataset_path("products")    # -> C:/Smart_shopping_EPA/data/raw/products.csv
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

# Project root = the folder that contains configs/ (works from scripts, notebooks and tests)
PROJECT_ROOT = Path(os.environ.get("SHOPSMART_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in config.yaml (raw_data, models, ...)."""
    return PROJECT_ROOT / CONFIG["paths"][key]


def dataset_info(name: str) -> dict:
    try:
        return CONFIG["datasets"][name]
    except KeyError:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(CONFIG['datasets'])}") from None


def dataset_path(name: str) -> Path:
    """Absolute path of a dataset file, e.g. dataset_path('interactions')."""
    return path("raw_data") / dataset_info(name)["file"]
