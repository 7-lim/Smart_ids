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
BALANCED_DIR  = DATA_DIR / "balanced"

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

# Classes with too few samples to learn (and that destabilise GAN training).
RARE_CLASSES_TO_DROP = [
    "Heartbleed",
    "Infiltration",
    "Web Attack \x96 Sql Injection",  # CICIDS uses the latin1 \x96 dash
]

LABEL_COL   = "Label"
ENCODED_COL = "label_encoded"

# --------------------------------------------------------------------------- #
# Training hyperparameters
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

MLP_CONFIG = {
    "hidden_units":   [256, 128, 64],
    "dropout":        0.3,
    "learning_rate":  1e-3,
    "batch_size":     1024,
    "epochs":         50,
    "patience":       7,
}

AE_CONFIG = {
    "encoder_units":  [64, 32, 16],
    "latent_dim":     8,
    "learning_rate":  1e-3,
    "batch_size":     512,
    "epochs":         50,
    "patience":       7,
    # Threshold is derived from the 99th-percentile of validation
    # reconstruction error on BENIGN traffic.
    "threshold_percentile": 99.0,
}

# Target per-class size after undersampling BENIGN before the GAN step.
UNDERSAMPLE_BENIGN_TO = 120_000
# Target size every minority class is grown to with CTGAN.
GAN_TARGET_PER_CLASS  = 50_000

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
    "Web Attack \x96 Brute Force": ("MEDIUM", "BLOCK_SOURCE_IP"),
    "Web Attack \x96 XSS":         ("HIGH",   "BLOCK_AND_INVESTIGATE"),
}
DEFAULT_POLICY = ("UNKNOWN", "ALERT_AND_MONITOR")

MODEL_VERSION = "1.0.0"
