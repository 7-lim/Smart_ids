"""
Plotting and metric helpers shared by the evaluation notebook.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def plot_class_distribution(
    series: pd.Series, title: str = "Class distribution", figsize=(12, 5)
):
    counts = series.value_counts()
    plt.figure(figsize=figsize)
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=60, ha="right")
    plt.title(title)
    plt.ylabel("count")
    plt.tight_layout()


def plot_confusion(
    y_true,
    y_pred,
    class_names: Sequence[str],
    normalize: bool = True,
    title: str = "Confusion matrix",
    figsize=(10, 8),
):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=False,
    )
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()


def text_report(y_true, y_pred, class_names: Sequence[str]) -> str:
    return classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )


def plot_reconstruction_error(
    err_benign,
    err_attack,
    threshold: float,
    title: str = "Autoencoder reconstruction error",
):
    plt.figure(figsize=(10, 5))
    sns.histplot(err_benign, bins=80, color="steelblue", label="BENIGN", stat="density")
    sns.histplot(err_attack, bins=80, color="crimson",   label="ATTACK", stat="density",
                 alpha=0.6)
    plt.axvline(threshold, color="black", ls="--", label=f"threshold={threshold:.4f}")
    plt.xlabel("reconstruction MSE")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def autoencoder_anomaly_metrics(err: np.ndarray, y_is_attack: np.ndarray, threshold: float):
    """Binary metrics for the AE used as 'is this an attack at all?'."""
    y_pred = (err > threshold).astype(int)
    tp = int(((y_pred == 1) & (y_is_attack == 1)).sum())
    fp = int(((y_pred == 1) & (y_is_attack == 0)).sum())
    fn = int(((y_pred == 0) & (y_is_attack == 1)).sum())
    tn = int(((y_pred == 0) & (y_is_attack == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    try:
        auc = roc_auc_score(y_is_attack, err)
    except ValueError:
        auc = float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "auc": auc,
    }
