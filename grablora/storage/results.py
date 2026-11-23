"""results/raw/<task>/<method>/<model>/seed_<x>.parquet

One file per (task, method, model, seed). Rows are per training step (split="train",
with loss/coverage) plus one final row (split="eval", with accuracy). The
identifier columns are repeated on every row so a directory of parquets concatenates
into one tidy dataframe with no path parsing.
"""
from __future__ import annotations

import os


def _slug(s: str) -> str:
    return s.replace("/", "--").replace(" ", "_")


def results_path(base: str, task: str, method: str, model: str, seed: int,
                 ext: str = "parquet") -> str:
    d = os.path.join(base, "raw", _slug(task), _slug(method), _slug(model))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"seed_{seed}.{ext}")


def save_results(df, path: str) -> str:
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)          # needs pyarrow
    elif path.endswith(".pkl"):
        df.to_pickle(path)
    else:
        raise ValueError(f"unknown extension for {path}")
    return path


def load_results(path: str):
    import pandas as pd
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_pickle(path)