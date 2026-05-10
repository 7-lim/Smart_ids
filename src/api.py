"""
FastAPI service wrapping `SmartTIDS_Predictor`.

Run locally:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1

Endpoints:
    GET  /health             -> liveness + model metadata
    GET  /features           -> the 77 feature names the model expects
    GET  /model/info         -> attack catalogue + version
    POST /predict            -> classify one flow {feature: value}
    POST /predict/batch      -> classify many flows in one call

The predictor (heavy: TF model + scaler + AE) is loaded **once** at startup
via the lifespan handler and reused by every request — important so the
cyber team can drive thousands of flows/sec through it.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, RootModel

from src import config as cfg
from src.inference import SmartTIDS_Predictor

logger = logging.getLogger("smarttids.api")

# --------------------------------------------------------------------------- #
# Schemas (Pydantic v2)
# --------------------------------------------------------------------------- #
class FlowInput(RootModel[Dict[str, float]]):
    """A single flow as `{feature_name: value}`. Missing keys default to 0."""


class BatchInput(BaseModel):
    flows: List[Dict[str, float]] = Field(
        ..., description="List of flow dicts; each is the same shape as POST /predict."
    )


class TopKItem(BaseModel):
    class_: str = Field(..., alias="class")
    probability: float

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    anomaly_threshold: float
    predicted_class: str
    predicted_class_id: int
    confidence: float
    top_3: List[Dict[str, Any]]
    severity: str
    recommended_action: str
    inference_time_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_features: int
    n_classes: int
    autoencoder_loaded: bool


class ModelInfo(BaseModel):
    model_version: str
    n_features: int
    n_classes: int
    classes: List[str]
    severity_policy: Dict[str, List[str]]   # class -> [severity, action]


# --------------------------------------------------------------------------- #
# Lifespan: load the predictor once
# --------------------------------------------------------------------------- #
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading SmartTIDS predictor…")
    state["predictor"] = SmartTIDS_Predictor()
    logger.info("Predictor ready (model_version=%s)", cfg.MODEL_VERSION)
    yield
    state.clear()


app = FastAPI(
    title="AI-SmartTIDS",
    version=cfg.MODEL_VERSION,
    description=(
        "Real-time intrusion-detection inference service. "
        "Classifies network flows into BENIGN or one of 11 attack families "
        "and returns a recommended action for the SOC."
    ),
    lifespan=lifespan,
)

# Permissive CORS — tighten in deployment if a single origin is known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _predictor() -> SmartTIDS_Predictor:
    p = state.get("predictor")
    if p is None:
        raise HTTPException(status_code=503, detail="predictor not loaded")
    return p


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    p = _predictor()
    return HealthResponse(
        status="ok",
        model_version=cfg.MODEL_VERSION,
        n_features=len(p.feature_names),
        n_classes=p.n_classes,
        autoencoder_loaded=p.autoencoder is not None,
    )


@app.get("/features", tags=["meta"])
def features():
    """Exact list of feature names (in order) the model expects."""
    return {"feature_names": _predictor().expected_features()}


@app.get("/model/info", response_model=ModelInfo, tags=["meta"])
def model_info():
    p = _predictor()
    return ModelInfo(
        model_version=cfg.MODEL_VERSION,
        n_features=len(p.feature_names),
        n_classes=p.n_classes,
        classes=[p.label_map[i] for i in sorted(p.label_map)],
        severity_policy={k: list(v) for k, v in cfg.ATTACK_POLICY.items()},
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(flow: FlowInput):
    """Classify a single flow."""
    try:
        return _predictor().predict_flow(flow.root)
    except Exception as exc:                                   # pragma: no cover
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=List[PredictionResponse],
          tags=["inference"])
def predict_batch(payload: BatchInput):
    """Classify many flows in one round trip."""
    if not payload.flows:
        return []
    try:
        return _predictor().predict_batch(payload.flows)
    except Exception as exc:                                   # pragma: no cover
        logger.exception("predict_batch failed")
        raise HTTPException(status_code=500, detail=str(exc))


# --------------------------------------------------------------------------- #
# CLI: python -m src.api  ->  start uvicorn on :8000
# --------------------------------------------------------------------------- #
def _run_dev():                                                # pragma: no cover
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":                                     # pragma: no cover
    _run_dev()
