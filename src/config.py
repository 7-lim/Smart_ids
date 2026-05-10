"""
Centralized configuration for the AI-SmartTIDS project.

All paths, hyperparameters, and thresholds live here so notebooks,
training scripts, and the production inference module stay in sync.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR      = PROJECT_ROOT / "data" / "cicids2017"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR  = DATA_DIR / "features"

MODELS_DIR    = PROJECT_ROOT / "models"

# Artifact filenames (kept here so inference.py never hard-codes paths).
SCALER_FILE       = MODELS_DIR / "scaler.joblib"
LABEL_MAP_FILE    = FEATURES_DIR / "label_map.json"
FEATURE_NAMES_FILE = FEATURES_DIR / "feature_names.json"
MLP_MODEL_FILE    = MODELS_DIR / "mlp_model.keras"
AE_MODEL_FILE     = MODELS_DIR / "autoencoder_model.keras"
AE_THRESHOLD_FILE = MODELS_DIR / "autoencoder_threshold.json"

# --------------------------------------------------------------------------- #
# Data cleaning
# --------------------------------------------------------------------------- #
# Identifier / leak columns dropped before training & inference.
COLS_TO_DROP = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp",
    "Source Port", "Destination Port", "Fwd Header Length.1",
]

# Classes with too few samples to learn.
# Note: the SQL Injection label as it appears in the raw CSV uses a latin-1
# 0x96 byte (an "en dash") between "Web Attack" and "Sql Injection". We
# match that byte exactly so the drop filter actually fires at preprocessing.
RARE_CLASSES_TO_DROP = [
    "Heartbleed",
    "Infiltration",
    "Web Attack \x96 Sql Injection",
]

LABEL_COL   = "Label"
ENCODED_COL = "label_encoded"

# --------------------------------------------------------------------------- #
# Training hyperparameters
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

MLP_CONFIG = {
    "hidden_units":   [512, 256, 128],
    "dropout":        0.3,
    "learning_rate":  1e-3,
    "batch_size":     2048,
    "epochs":         60,
    "patience":       12,
    # Imbalance strategy. Options:
    #   "class_weights" : sklearn balanced weights passed to model.fit
    #   "focal"         : focal loss, no class weights
    #   "none"          : vanilla cross-entropy (will collapse to BENIGN)
    "imbalance_strategy": "class_weights",
    # Focal-loss hyperparameters (only used when imbalance_strategy == "focal").
    "focal_gamma":    2.0,
    "focal_alpha":    0.25,
    # Cap class-weights so the rarest classes don't dominate gradients.
    # 10 is a good middle ground: rare classes still get ~100x more importance
    # than BENIGN, but not so much that the model over-predicts them.
    "max_class_weight": 10.0,
}

AE_CONFIG = {
    "encoder_units":  [128, 64, 32],
    "latent_dim":     6,
    "learning_rate":  5e-4,
    "batch_size":     512,
    "epochs":         50,
    "patience":       10,
    # Default percentile used when starting threshold; the notebook then
    # sweeps over [90, 95, 97, 99, 99.5, 99.9] and picks the best F1.
    "threshold_percentile": 95.0,
    # Cap (in standard-deviations of the scaled features) used during
    # AE training to drop benign outliers that otherwise dominate MSE
    # and prevent the AE from learning the bulk of typical traffic.
    # Inference does NOT clip: extreme values then become anomaly signals.
    "outlier_clip_z": 5.0,
}

# --------------------------------------------------------------------------- #
# Inference / response policy
# --------------------------------------------------------------------------- #
# Mapping from attack family -> (severity, recommended_action).
# Used by SmartTIDS_Predictor to enrich responses for the cyber team.
ATTACK_POLICY = {
    "BENIGN":                    ("INFO",     "ALLOW"),
    "Bot":                       ("HIGH",     "BLOCK_AND_INVESTIGATE"),
    "DDoS":                      ("CRITICAL", "BLOCK_IMMEDIATELY"),
    "DoS GoldenEye":             ("HIGH",     "RATE_LIMIT_AND_ALERT"),
    "DoS Hulk":                  ("HIGH",     "RATE_LIMIT_AND_ALERT"),
    "DoS Slowhttptest":          ("HIGH",     "RATE_LIMIT_AND_ALERT"),
    "DoS slowloris":             ("HIGH",     "RATE_LIMIT_AND_ALERT"),
    "FTP-Patator":               ("MEDIUM",   "BLOCK_SOURCE_IP"),
    "SSH-Patator":               ("MEDIUM",   "BLOCK_SOURCE_IP"),
    "PortScan":                  ("MEDIUM",   "ALERT_AND_MONITOR"),
    "Web Attack - Brute Force":  ("MEDIUM", "BLOCK_SOURCE_IP"),
    "Web Attack - XSS":          ("HIGH",   "BLOCK_AND_INVESTIGATE"),
}
DEFAULT_POLICY = ("UNKNOWN", "ALERT_AND_MONITOR")

MODEL_VERSION = "1.0.0"
