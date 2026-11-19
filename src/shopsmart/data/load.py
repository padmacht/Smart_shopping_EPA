"""
Load datasets by name using the map in configs/config.yaml.

    from shopsmart.data import load, load_all, list_datasets
    products = load("products")
    data = load_all()                 # dict of all datasets (except evaluation-only ones)
    list_datasets()                   # table of names, files, rows, descriptions
"""
from __future__ import annotations

import pandas as pd

from shopsmart.config import CONFIG, dataset_info, dataset_path


def load(name: str, nrows: int | None = None) -> pd.DataFrame:
    """Read one dataset with its dates parsed and nullable integer columns kept as integers."""
    info = dataset_info(name)
    file = dataset_path(name)
    if not file.exists():
        raise FileNotFoundError(f"{file} not found. Put the CSV files in data/raw (see README).")
    df = pd.read_csv(file, nrows=nrows)
    for col in info.get("parse_dates", []):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    # these columns are only filled on some event types -> keep as nullable integers
    for col in ("Rating", "Quantity"):
        if col in df.columns and name == "interactions":
            df[col] = df[col].astype("Int64")
    return df


def load_all(include_evaluation_only: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for name, info in CONFIG["datasets"].items():
        if info.get("evaluation_only") and not include_evaluation_only:
            continue
        out[name] = load(name)
    return out


def list_datasets() -> pd.DataFrame:
    rows = []
    for name, info in CONFIG["datasets"].items():
        f = dataset_path(name)
        n = sum(1 for _ in open(f, encoding="utf-8")) - 1 if f.exists() else None
        rows.append({"name": name, "file": info["file"], "rows": n, "exists": f.exists(),
                     "evaluation_only": info.get("evaluation_only", False), "description": info["description"]})
    return pd.DataFrame(rows).set_index("name")
