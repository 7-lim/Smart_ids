"""
SmartTIDS_Predictor — production-ready inference for AI-SmartTIDS.

Designed to be the single integration point for the cyber-security team.

Quick start
-----------
    from src.inference import SmartTIDS_Predictor

    predictor = SmartTIDS_Predictor()          # auto-loads everything

    # Either form works:
    result = predictor.predict_flow(flow_dict)        # one flow as a dict
    result = predictor.predict(features_array)        # numpy / list
    results = predictor.predict_batch(df_or_array)    # bulk

The hybrid decision policy
--------------------------
* The Autoencoder gives a reconstruction error -> `is_anomaly` flag.
* The MLP always returns class probabilities.
* Final label:
    - if MLP confidently says BENIGN AND AE says normal  -> BENIGN
    - if AE flags anomaly AND MLP says BENIGN            -> UNKNOWN_ANOMALY
      (likely zero-day; raises severity)
    - otherwise                                          -> MLP top class
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src import config as cfg
from src.preprocessing import align_features

logger = logging.getLogger("smarttids.inference")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


FlowLike = Union[Mapping[str, Any], Sequence[float], np.ndarray, pd.Series]


# --------------------------------------------------------------------------- #
# Result schema
# --------------------------------------------------------------------------- #
@dataclass
class PredictionResult:
    """Structured prediction. Convert to dict via `.to_dict()` for JSON APIs."""
    is_anomaly: bool
    anomaly_score: float                 # AE reconstruction MSE
    anomaly_threshold: float
    predicted_class: str
    predicted_class_id: int
    confidence: float                    # MLP top-1 probability
    top_3: List[Dict[str, float]]        # [{"class": "...", "probability": ...}, ...]
    severity: str                        # INFO | MEDIUM | HIGH | CRITICAL | UNKNOWN
    recommended_action: str              # ALLOW | RATE_LIMIT_AND_ALERT | BLOCK_* | ...
    inference_time_ms: float
    model_version: str = cfg.MODEL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly":          self.is_anomaly,
            "anomaly_score":       round(self.anomaly_score, 6),
            "anomaly_threshold":   round(self.anomaly_threshold, 6),
            "predicted_class":     self.predicted_class,
            "predicted_class_id":  self.predicted_class_id,
            "confidence":          round(self.confidence, 4),
            "top_3":               self.top_3,
            "severity":            self.severity,
            "recommended_action":  self.recommended_action,
            "inference_time_ms":   round(self.inference_time_ms, 3),
            "model_version":       self.model_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# --------------------------------------------------------------------------- #
# The predictor
# --------------------------------------------------------------------------- #
class SmartTIDS_Predictor:
    """
    Loads the MLP, the Autoencoder, the scaler, and the label map once,
    then serves single-flow or batch predictions.

    Parameters
    ----------
    models_dir, features_dir
        Override default artifact locations (useful in tests).
    confidence_threshold
        MLP probability below which a prediction is downgraded to
        "UNKNOWN_ANOMALY" (default 0.4).
    use_autoencoder
        If False, skip AE entirely and rely on the MLP only.
    """

    UNKNOWN_LABEL = "UNKNOWN_ANOMALY"

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        features_dir: Optional[Path] = None,
        confidence_threshold: float = 0.40,
        use_autoencoder: bool = True,
    ):
        self.models_dir   = Path(models_dir or cfg.MODELS_DIR)
        self.features_dir = Path(features_dir or cfg.FEATURES_DIR)
        self.confidence_threshold = float(confidence_threshold)
        self.use_autoencoder = bool(use_autoencoder)

        self._load_artifacts()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_artifacts(self) -> None:
        import joblib

        # Defer the TF import so importing this module is cheap if the
        # caller only needs the policy/utilities.
        from tensorflow.keras.models import load_model

        scaler_path  = self.models_dir / cfg.SCALER_FILE.name
        mlp_path     = self.models_dir / cfg.MLP_MODEL_FILE.name
        ae_path      = self.models_dir / cfg.AE_MODEL_FILE.name
        thresh_path  = self.models_dir / cfg.AE_THRESHOLD_FILE.name
        labels_path  = self.features_dir / cfg.LABEL_MAP_FILE.name
        feats_path   = self.features_dir / cfg.FEATURE_NAMES_FILE.name

        for p in (scaler_path, mlp_path, labels_path, feats_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required artifact missing: {p}. "
                    f"Train the model first (see notebooks 01-04)."
                )

        logger.info("Loading scaler: %s", scaler_path)
        self.scaler = joblib.load(scaler_path)

        logger.info("Loading feature names: %s", feats_path)
        with open(feats_path, encoding="utf-8") as f:
            self.feature_names: List[str] = json.load(f)

        logger.info("Loading label map: %s", labels_path)
        with open(labels_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.label_map: Dict[int, str] = {int(k): v for k, v in raw.items()}
        self.n_classes = len(self.label_map)

        logger.info("Loading MLP: %s", mlp_path)
        self.mlp = load_model(mlp_path, compile=False)

        if self.use_autoencoder and ae_path.exists():
            logger.info("Loading autoencoder: %s", ae_path)
            self.autoencoder = load_model(ae_path, compile=False)
            if thresh_path.exists():
                self.ae_threshold = float(json.loads(thresh_path.read_text())["threshold"])
            else:
                logger.warning("AE threshold file missing; using fallback 0.01")
                self.ae_threshold = 0.01
        else:
            self.autoencoder = None
            self.ae_threshold = float("inf")
            if self.use_autoencoder:
                logger.warning(
                    "Autoencoder requested but %s not found; running MLP-only.",
                    ae_path,
                )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict_flow(self, flow: Mapping[str, Any]) -> Dict[str, Any]:
        """Predict one flow given as a dict of {feature_name: value}."""
        result = self._predict_one(flow)
        return result.to_dict()

    def predict(self, features: FlowLike) -> Dict[str, Any]:
        """Predict one flow given as dict, list, or 1-D array of features."""
        result = self._predict_one(features)
        return result.to_dict()

    def predict_batch(
        self, flows: Union[pd.DataFrame, np.ndarray, Iterable[Mapping]],
    ) -> List[Dict[str, Any]]:
        """Predict a batch of flows. Returns a list of result dicts."""
        df = self._coerce_to_df(flows)
        if len(df) == 0:
            return []
        X = self._scale(df)

        t0 = time.perf_counter()
        proba = self.mlp.predict(X, verbose=0)
        if self.autoencoder is not None:
            recon = self.autoencoder.predict(X, verbose=0)
            err   = np.mean((X - recon) ** 2, axis=1)
        else:
            err = np.zeros(len(X), dtype=float)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        per_row_ms = elapsed_ms / max(len(X), 1)
        return [
            self._compose_result(proba[i], float(err[i]), per_row_ms).to_dict()
            for i in range(len(X))
        ]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _predict_one(self, flow: FlowLike) -> PredictionResult:
        df = self._coerce_to_df([flow] if not isinstance(flow, pd.DataFrame) else flow)
        X  = self._scale(df)

        t0 = time.perf_counter()
        proba = self.mlp.predict(X, verbose=0)[0]
        if self.autoencoder is not None:
            recon = self.autoencoder.predict(X, verbose=0)
            err = float(np.mean((X - recon) ** 2, axis=1)[0])
        else:
            err = 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return self._compose_result(proba, err, elapsed_ms)

    def _coerce_to_df(self, flows) -> pd.DataFrame:
        """Accept dict / list-of-dicts / array / DataFrame; return aligned DF."""
        if isinstance(flows, pd.DataFrame):
            df = flows
        elif isinstance(flows, Mapping):
            df = pd.DataFrame([dict(flows)])
        elif isinstance(flows, (list, tuple)) and flows and isinstance(flows[0], Mapping):
            df = pd.DataFrame([dict(f) for f in flows])
        else:
            arr = np.asarray(flows, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Array input has {arr.shape[1]} features, "
                    f"expected {len(self.feature_names)}. "
                    f"Pass a dict if you cannot guarantee column order."
                )
            df = pd.DataFrame(arr, columns=self.feature_names)
        return align_features(df, self.feature_names)

    def _scale(self, df: pd.DataFrame) -> np.ndarray:
        # Replace inf with NaN -> 0 to mirror training cleanup.
        X = df.to_numpy(dtype=np.float32, copy=True)
        X[~np.isfinite(X)] = 0.0
        return self.scaler.transform(X)

    def _compose_result(
        self, proba: np.ndarray, recon_err: float, elapsed_ms: float
    ) -> PredictionResult:
        proba = np.asarray(proba).ravel()
        top1_id = int(np.argmax(proba))
        top1_label = self.label_map[top1_id]
        confidence = float(proba[top1_id])

        # Top-3 (or fewer if classifier has <3 classes).
        order = np.argsort(proba)[::-1][: min(3, len(proba))]
        top_3 = [
            {"class": self.label_map[int(i)], "probability": round(float(proba[i]), 4)}
            for i in order
        ]

        is_anomaly = bool(recon_err > self.ae_threshold)

        # Hybrid decision policy.
        if top1_label == "BENIGN" and is_anomaly:
            label = self.UNKNOWN_LABEL
            severity, action = "HIGH", "ALERT_AND_INVESTIGATE"
        elif confidence < self.confidence_threshold and is_anomaly:
            label = self.UNKNOWN_LABEL
            severity, action = "HIGH", "ALERT_AND_INVESTIGATE"
        else:
            label = top1_label
            severity, action = cfg.ATTACK_POLICY.get(label, cfg.DEFAULT_POLICY)
            # Re-classify BENIGN with low confidence as suspicious.
            if label == "BENIGN" and confidence < self.confidence_threshold:
                severity, action = "MEDIUM", "MONITOR"

        return PredictionResult(
            is_anomaly=is_anomaly,
            anomaly_score=float(recon_err),
            anomaly_threshold=float(self.ae_threshold),
            predicted_class=label,
            predicted_class_id=top1_id if label != self.UNKNOWN_LABEL else -1,
            confidence=confidence,
            top_3=top_3,
            severity=severity,
            recommended_action=action,
            inference_time_ms=float(elapsed_ms),
        )

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def expected_features(self) -> List[str]:
        """The exact feature names (in order) the predictor expects."""
        return list(self.feature_names)

    def healthcheck(self) -> Dict[str, Any]:
        """Quick smoke-test result, useful for /health endpoints."""
        dummy = {name: 0.0 for name in self.feature_names}
        out = self.predict_flow(dummy)
        return {
            "status": "ok",
            "model_version": cfg.MODEL_VERSION,
            "n_features": len(self.feature_names),
            "n_classes": self.n_classes,
            "autoencoder_loaded": self.autoencoder is not None,
            "sample_prediction": out,
        }


# --------------------------------------------------------------------------- #
# CLI for ops smoke-tests:  python -m src.inference --healthcheck
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartTIDS inference utility")
    parser.add_argument("--healthcheck", action="store_true",
                        help="Load all artifacts and run a dummy prediction.")
    parser.add_argument("--no-autoencoder", action="store_true",
                        help="Skip loading the autoencoder.")
    args = parser.parse_args()

    p = SmartTIDS_Predictor(use_autoencoder=not args.no_autoencoder)
    if args.healthcheck:
        print(json.dumps(p.healthcheck(), indent=2))
