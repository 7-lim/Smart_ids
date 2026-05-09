"""
Reusable preprocessing primitives for AI-SmartTIDS.

The exact same logic used to clean the training corpus is also called by
`SmartTIDS_Predictor` at inference time, so a flow processed at runtime
sees the same column order, dtypes, and scaling as the model was trained on.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import config as cfg


# --------------------------------------------------------------------------- #
# Loading raw CICIDS2017 CSVs
# --------------------------------------------------------------------------- #
def _read_csv_robust(path: str | Path) -> pd.DataFrame:
    """Read a CICIDS CSV with the encoding/parser that actually works."""
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1", low_memory=False)
        except Exception:
            return pd.read_csv(
                path, encoding="latin1", engine="python", on_bad_lines="skip"
            )


def load_raw_directory(raw_dir: str | Path = cfg.RAW_DIR) -> pd.DataFrame:
    """Load and concatenate every CSV in `raw_dir`, tagging the source file."""
    raw_dir = Path(raw_dir)
    frames = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        df = _read_csv_robust(raw_dir / fname)
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        df["source_file"] = fname
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop NaN/Inf rows and identifier columns. Idempotent — safe to call
    on already-clean frames.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Flow Bytes/s" in df.columns:
        df["Flow Bytes/s"] = pd.to_numeric(df["Flow Bytes/s"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    drop = [c for c in cfg.COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop)

    return df.reset_index(drop=True)


def drop_rare_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove classes that are too small to learn."""
    if cfg.LABEL_COL not in df.columns:
        return df
    mask = ~df[cfg.LABEL_COL].isin(cfg.RARE_CLASSES_TO_DROP)
    return df.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature alignment (the inference contract)
# --------------------------------------------------------------------------- #
def align_features(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """
    Force `df` to have exactly `feature_names` as columns, in that order.

    Missing columns are filled with 0; extra columns are dropped. This
    guarantees that whatever a caller hands us at runtime can still be
    fed into a model trained on a known schema.
    """
    feature_names = list(feature_names)
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_names]
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


# --------------------------------------------------------------------------- #
# Artifact IO
# --------------------------------------------------------------------------- #
def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
