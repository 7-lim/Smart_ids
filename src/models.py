"""
Keras model builders for AI-SmartTIDS.

Two architectures are exposed:

* `build_mlp`         — supervised multi-class classifier over attack families
* `build_autoencoder` — unsupervised anomaly detector trained on BENIGN only

Keeping these in a single module means notebooks and the inference layer
can both rebuild a model from the same source of truth (useful when
loading legacy weights).
"""
from __future__ import annotations

from typing import List

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_mlp(
    input_dim: int,
    n_classes: int,
    hidden_units: List[int] = (256, 128, 64),
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Plain MLP classifier with batch-norm + dropout regularisation."""
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="probs")(x)

    model = models.Model(inputs, outputs, name="smarttids_mlp")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_autoencoder(
    input_dim: int,
    encoder_units: List[int] = (64, 32, 16),
    latent_dim: int = 8,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Symmetric tabular autoencoder.

    Trained only on BENIGN traffic so reconstruction error spikes for any
    flow whose statistics deviate from normal — useful as a zero-day net.
    """
    inputs = layers.Input(shape=(input_dim,), name="features")

    x = inputs
    for i, units in enumerate(encoder_units):
        x = layers.Dense(units, activation="relu", name=f"enc_{i}")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    x = latent
    for i, units in enumerate(reversed(encoder_units)):
        x = layers.Dense(units, activation="relu", name=f"dec_{i}")(x)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    model = models.Model(inputs, outputs, name="smarttids_autoencoder")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def reconstruction_error(model: tf.keras.Model, X) -> "np.ndarray":
    """Per-row MSE between input and reconstruction."""
    import numpy as np
    recon = model.predict(X, verbose=0)
    return np.mean((X - recon) ** 2, axis=1)
