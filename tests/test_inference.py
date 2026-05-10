"""
Tests for src.inference.SmartTIDS_Predictor.

These exercise the **integration contract** the cyber team relies on:

* artifacts load, healthcheck succeeds
* dict / list / array / DataFrame inputs all work
* missing or extra features are tolerated
* batch and single predictions agree
* the JSON response always carries the expected keys

Run with:
    pytest tests/

Skipped automatically if the trained artifacts (mlp_model.keras / scaler /
autoencoder) aren't present yet — handy on fresh checkouts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as cfg


# Skip the whole module if training hasn't been run.
_ARTIFACTS_READY = (
    cfg.SCALER_FILE.exists()
    and cfg.MLP_MODEL_FILE.exists()
    and cfg.LABEL_MAP_FILE.exists()
    and cfg.FEATURE_NAMES_FILE.exists()
)
pytestmark = pytest.mark.skipif(
    not _ARTIFACTS_READY,
    reason="Trained artifacts missing — run notebooks 01-03 first.",
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def predictor():
    from src.inference import SmartTIDS_Predictor
    return SmartTIDS_Predictor()


@pytest.fixture(scope="module")
def feature_names(predictor):
    return predictor.expected_features()


@pytest.fixture
def sample_flow(feature_names):
    """A zero-valued flow — all 77 keys present, all values 0."""
    return {name: 0.0 for name in feature_names}


# Required keys in every PredictionResult.to_dict().
_REQUIRED = {
    "is_anomaly", "anomaly_score", "anomaly_threshold",
    "predicted_class", "predicted_class_id", "confidence",
    "top_3", "severity", "recommended_action",
    "inference_time_ms", "model_version",
}


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def test_loads_all_artifacts(predictor):
    assert predictor.scaler is not None
    assert predictor.mlp is not None
    assert predictor.feature_names, "feature_names is empty"
    assert predictor.label_map, "label_map is empty"
    # AE is optional — if it loaded, threshold must be a finite float.
    if predictor.autoencoder is not None:
        assert np.isfinite(predictor.ae_threshold)


def test_healthcheck_returns_ok(predictor):
    h = predictor.healthcheck()
    assert h["status"] == "ok"
    assert h["model_version"] == cfg.MODEL_VERSION
    assert h["n_features"] == len(predictor.feature_names)
    assert h["n_classes"] == len(predictor.label_map)
    assert _REQUIRED.issubset(h["sample_prediction"].keys())


# --------------------------------------------------------------------------- #
# 2. Input shapes
# --------------------------------------------------------------------------- #
def test_predict_dict_input(predictor, sample_flow):
    r = predictor.predict_flow(sample_flow)
    assert _REQUIRED.issubset(r.keys())
    assert 0.0 <= r["confidence"] <= 1.0
    assert isinstance(r["is_anomaly"], bool)


def test_predict_array_input(predictor, feature_names):
    arr = np.zeros(len(feature_names), dtype=np.float32)
    r = predictor.predict(arr)
    assert _REQUIRED.issubset(r.keys())


def test_predict_array_wrong_length_raises(predictor, feature_names):
    bad = np.zeros(len(feature_names) - 1)
    with pytest.raises(ValueError, match="features"):
        predictor.predict(bad)


def test_predict_batch_dataframe(predictor, sample_flow):
    df = pd.DataFrame([sample_flow] * 5)
    results = predictor.predict_batch(df)
    assert len(results) == 5
    assert all(_REQUIRED.issubset(r.keys()) for r in results)


def test_predict_batch_list_of_dicts(predictor, sample_flow):
    results = predictor.predict_batch([sample_flow, sample_flow])
    assert len(results) == 2


def test_predict_batch_empty_dataframe_does_not_crash(predictor, feature_names):
    df = pd.DataFrame(columns=feature_names)
    # Empty input is unusual but should not raise an exception.
    results = predictor.predict_batch(df)
    assert results == []


# --------------------------------------------------------------------------- #
# 3. Schema flexibility — the cyber team's flows won't always be perfect
# --------------------------------------------------------------------------- #
def test_missing_features_zero_filled(predictor):
    # Only two known features supplied — predictor must not error.
    partial = {"Flow Duration": 1000.0, "Total Fwd Packets": 5.0}
    r = predictor.predict_flow(partial)
    assert _REQUIRED.issubset(r.keys())


def test_extra_features_dropped(predictor, sample_flow):
    polluted = dict(sample_flow)
    polluted["completely_made_up_column"] = 42.0
    polluted["another_garbage_field"] = "not even a number"
    r = predictor.predict_flow(polluted)
    assert _REQUIRED.issubset(r.keys())


# --------------------------------------------------------------------------- #
# 4. Response semantics
# --------------------------------------------------------------------------- #
def test_top3_is_well_formed(predictor, sample_flow):
    r = predictor.predict_flow(sample_flow)
    top3 = r["top_3"]
    assert 1 <= len(top3) <= 3
    for item in top3:
        assert "class" in item
        assert "probability" in item
        assert 0.0 <= item["probability"] <= 1.0
    # Top-1 from top_3 should match the predicted_class (when not UNKNOWN).
    if r["predicted_class"] != "UNKNOWN_ANOMALY":
        assert top3[0]["class"] == r["predicted_class"]


def test_severity_is_known(predictor, sample_flow):
    r = predictor.predict_flow(sample_flow)
    assert r["severity"] in {
        "INFO", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN",
    }


def test_response_is_json_serialisable(predictor, sample_flow):
    r = predictor.predict_flow(sample_flow)
    # Must round-trip cleanly so the API can return it.
    assert json.loads(json.dumps(r)) == r


def test_batch_and_single_agree(predictor, sample_flow):
    """Same input, two paths -> same predicted class."""
    single = predictor.predict_flow(sample_flow)
    batch  = predictor.predict_batch([sample_flow])[0]
    assert single["predicted_class"] == batch["predicted_class"]
    # Confidence should be very close (timing field will differ).
    assert abs(single["confidence"] - batch["confidence"]) < 1e-4
