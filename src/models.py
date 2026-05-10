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


# --------------------------------------------------------------------------- #
# Loss functions for imbalanced classification
# --------------------------------------------------------------------------- #
def sparse_categorical_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal loss for sparse integer labels.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    `gamma` down-weights easy examples (high p_t); `alpha` is a global
    rebalancing scalar. Use this when class imbalance is severe but you do
    not want to pass `class_weight` at fit time.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Gather the predicted probability of the true class for each row.
        idx = tf.stack(
            [tf.range(tf.shape(y_true)[0]), tf.reshape(y_true, [-1])], axis=1
        )
        p_t = tf.gather_nd(y_pred, idx)

        focal = -alpha * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal)

    loss.__name__ = "sparse_categorical_focal_loss"
    return loss


# --------------------------------------------------------------------------- #
# MLP classifier
# --------------------------------------------------------------------------- #
def build_mlp(
    input_dim: int,
    n_classes: int,
    hidden_units: List[int] = (256, 128, 64),
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    loss: str = "sparse_categorical_crossentropy",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> tf.keras.Model:
    """
    Plain MLP classifier with batch-norm + dropout regularisation.

    Parameters
    ----------
    loss : str
        One of ``"sparse_categorical_crossentropy"`` (default, pair with
        ``class_weight`` at fit time) or ``"focal"`` (focal loss, no
        class-weight needed).
    """
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="probs")(x)

    if loss == "focal":
        loss_fn = sparse_categorical_focal_loss(focal_gamma, focal_alpha)
    else:
        loss_fn = "sparse_categorical_crossentropy"

    model = models.Model(inputs, outputs, name="smarttids_mlp")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# Autoencoder (anomaly detector)
# --------------------------------------------------------------------------- #
def build_autoencoder(
    input_dim: int,
    encoder_units: List[int] = (128, 64, 32),
    latent_dim: int = 6,
    learning_rate: float = 5e-4,
) -> tf.keras.Model:
    """
    Symmetric tabular autoencoder.

    Trained only on BENIGN traffic so reconstruction error spikes for any
    flow whose statistics deviate from normal — useful as a zero-day net.

    Each Dense layer is followed by BatchNorm + LeakyReLU; LeakyReLU avoids
    the "dying ReLU" problem common with the sparse, heavy-tailed numeric
    features in CICIDS. The latent layer has linear activation so the
    bottleneck can represent both positive and negative deviations.
    """
    inputs = layers.Input(shape=(input_dim,), name="features")

    x = inputs
    for i, units in enumerate(encoder_units):
        x = layers.Dense(units, name=f"enc_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
        x = layers.LeakyReLU(negative_slope=0.1, name=f"enc_act_{i}")(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    x = latent
    for i, units in enumerate(reversed(encoder_units)):
        x = layers.Dense(units, name=f"dec_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)
        x = layers.LeakyReLU(negative_slope=0.1, name=f"dec_act_{i}")(x)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    model = models.Model(inputs, outputs, name="smarttids_autoencoder")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def reconstruction_error(model: tf.keras.Model, X) -> "np.ndarray":
    """Per-row MSE between input and reconstruction."""
    import numpy as np
    recon = model.predict(X, verbose=0)
    return np.mean((X - recon) ** 2, axis=1)


def compute_balanced_class_weights(
    y, max_weight: float = 50.0
) -> dict:
    """
    sklearn-balanced class weights, capped at `max_weight` so a class that's
    1000x rarer than BENIGN doesn't completely dominate gradients.
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    raw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    capped = np.minimum(raw, max_weight)
    return {int(c): float(w) for c, w in zip(classes, capped)}
