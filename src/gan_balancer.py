"""
CTGAN-based class balancer.

Two-stage strategy used in `notebooks/02_data_balancing_gan.ipynb`:

1. Undersample the BENIGN class with `RandomUnderSampler`.
2. Fit one CTGAN per minority class on its real samples and synthesise
   enough rows to reach `target_per_class`.

CTGAN handles tabular distributions better than vanilla SMOTE for the
heavy-tailed numeric features in CICIDS2017 (Flow Bytes/s, IAT, ...).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src import config as cfg


def undersample_benign(
    df: pd.DataFrame,
    target_size: int = cfg.UNDERSAMPLE_BENIGN_TO,
    label_col: str = cfg.LABEL_COL,
    random_state: int = cfg.RANDOM_STATE,
) -> pd.DataFrame:
    """Reduce the BENIGN class to `target_size`. Other classes untouched."""
    benign = df[df[label_col] == "BENIGN"]
    other  = df[df[label_col] != "BENIGN"]
    if len(benign) > target_size:
        benign = benign.sample(n=target_size, random_state=random_state)
    return pd.concat([benign, other], ignore_index=True).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)


def gan_oversample(
    df: pd.DataFrame,
    target_per_class: int = cfg.GAN_TARGET_PER_CLASS,
    label_col: str = cfg.LABEL_COL,
    epochs: int = 150,
    skip_classes: tuple = ("BENIGN",),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For every class with fewer than `target_per_class` rows, fit a CTGAN
    and synthesise the gap.

    CTGAN is imported lazily so the rest of the package does not require
    the heavy dependency just to run inference.
    """
    try:
        from ctgan import CTGAN
    except ImportError as exc:
        raise ImportError(
            "CTGAN is required for gan_oversample. Install with: "
            "pip install ctgan"
        ) from exc

    pieces = []
    counts: Dict[str, int] = df[label_col].value_counts().to_dict()

    for cls, n in counts.items():
        cls_df = df[df[label_col] == cls]

        if cls in skip_classes or n >= target_per_class:
            pieces.append(cls_df)
            if verbose:
                print(f"[skip] {cls}: {n:,} rows (no synthesis)")
            continue

        # Fit a CTGAN on this class only — features minus the label.
        feature_df = cls_df.drop(columns=[label_col])
        gan = CTGAN(epochs=epochs, verbose=False)
        gan.fit(feature_df)

        n_to_make = target_per_class - n
        synth = gan.sample(n_to_make)
        synth[label_col] = cls

        if verbose:
            print(f"[gan]  {cls}: {n:,} -> {target_per_class:,} (+{n_to_make:,})")

        pieces.append(pd.concat([cls_df, synth], ignore_index=True))

    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=cfg.RANDOM_STATE)
        .reset_index(drop=True)
    )
