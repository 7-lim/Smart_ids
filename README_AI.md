# AI-SmartTIDS — AI Module

Intelligent real-time Intrusion Detection System for the **AI-SmartTIDS** PFA project.
This document is written for the **cyber-security integration team** — it explains
how to install the AI module, what it produces, and how to call it from your code.

---

## 1. Project layout

```
smart_ids/
├── data/cicids2017/
│   ├── raw/                   # original CICIDS2017 CSVs (you provide)
│   ├── processed/             # train/val/test splits (built by notebook 01)
│   ├── features/              # label_map.json, feature_names.json
│   └── balanced/              # CTGAN-balanced training set (notebook 02)
├── models/                    # trained artifacts (built by notebooks 03 & 04)
│   ├── scaler.joblib
│   ├── mlp_model.keras
│   ├── autoencoder_model.keras
│   └── autoencoder_threshold.json
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_data_balancing_gan.ipynb
│   ├── 03_mlp_training.ipynb
│   ├── 04_autoencoder_training.ipynb
│   └── 05_evaluation_and_comparison.ipynb
├── src/
│   ├── config.py              # all paths, hyperparams, attack policy
│   ├── preprocessing.py       # cleaning + feature alignment (shared with inference)
│   ├── models.py              # Keras MLP & autoencoder builders
│   ├── gan_balancer.py        # CTGAN oversampling
│   ├── evaluation.py          # plotting / metric helpers
│   └── inference.py           # ★ SmartTIDS_Predictor — production entry point
├── requirements.txt
└── README_AI.md
```

---

## 2. Installation

```powershell
# from the project root (Windows / PowerShell)
python -m venv ids_pfa
.\ids_pfa\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU optional. CTGAN (notebook 02 only) requires `pytorch`; install separately
if you want to retrain: `pip install torch`.

---

## 3. Training pipeline (run once)

```bash
jupyter notebook notebooks/01_eda_and_preprocessing.ipynb   # cleans + splits
jupyter notebook notebooks/02_data_balancing_gan.ipynb      # balances minorities
jupyter notebook notebooks/03_mlp_training.ipynb            # trains MLP
jupyter notebook notebooks/04_autoencoder_training.ipynb    # trains AE
jupyter notebook notebooks/05_evaluation_and_comparison.ipynb
```

After notebooks 01-04 the `models/` directory contains everything the inference
module needs. Retraining is **not required** to use the API.

---

## 4. Using the inference API (cyber-team integration)

### 4.1 Basic — one flow at a time

```python
from src.inference import SmartTIDS_Predictor

predictor = SmartTIDS_Predictor()      # loads scaler + MLP + AE + label map

flow = {
    "Flow Duration":           5_120_343,
    "Total Fwd Packets":       12,
    "Total Backward Packets":  10,
    "Flow Bytes/s":            1023.4,
    # ... any subset of the 77 CICIDS2017 features
}

result = predictor.predict_flow(flow)
print(result)
```

Example response:

```json
{
  "is_anomaly": true,
  "anomaly_score": 0.0823,
  "anomaly_threshold": 0.0124,
  "predicted_class": "DDoS",
  "predicted_class_id": 2,
  "confidence": 0.9731,
  "top_3": [
    {"class": "DDoS",      "probability": 0.9731},
    {"class": "DoS Hulk",  "probability": 0.0211},
    {"class": "PortScan",  "probability": 0.0035}
  ],
  "severity": "CRITICAL",
  "recommended_action": "BLOCK_IMMEDIATELY",
  "inference_time_ms": 4.2,
  "model_version": "1.0.0"
}
```

### 4.2 Batch (recommended for high throughput)

```python
import pandas as pd

flows_df = pd.read_csv("incoming_flows.csv")
results  = predictor.predict_batch(flows_df)         # list[dict]
```

### 4.3 Numpy array

```python
import numpy as np
arr = np.zeros(len(predictor.expected_features()))
predictor.predict(arr)        # works the same way
```

### 4.4 Health-check endpoint

```python
predictor.healthcheck()
# {'status': 'ok', 'model_version': '1.0.0', 'n_features': 77,
#  'n_classes': 12, 'autoencoder_loaded': True, 'sample_prediction': {...}}
```

You can also run it from the shell:

```bash
python -m src.inference --healthcheck
```

---

## 5. Response schema reference

| Field | Type | Meaning |
|---|---|---|
| `is_anomaly` | bool | AE reconstruction error exceeded threshold |
| `anomaly_score` | float | Reconstruction MSE for the flow |
| `anomaly_threshold` | float | Static threshold loaded from `autoencoder_threshold.json` |
| `predicted_class` | str | `"BENIGN"`, an attack family, or `"UNKNOWN_ANOMALY"` |
| `predicted_class_id` | int | Class index, `-1` for `UNKNOWN_ANOMALY` |
| `confidence` | float | MLP top-1 probability |
| `top_3` | list[obj] | Top-3 candidate classes with probabilities |
| `severity` | str | `INFO` / `MEDIUM` / `HIGH` / `CRITICAL` / `UNKNOWN` |
| `recommended_action` | str | One of `ALLOW`, `MONITOR`, `ALERT_AND_MONITOR`, `RATE_LIMIT_AND_ALERT`, `BLOCK_SOURCE_IP`, `BLOCK_AND_INVESTIGATE`, `ALERT_AND_INVESTIGATE`, `BLOCK_IMMEDIATELY` |
| `inference_time_ms` | float | Wall-clock latency for this prediction |
| `model_version` | str | Tied to artifacts under `models/` |

---

## 6. Hybrid decision logic

The predictor combines two independent signals:

1. **MLP** — supervised classifier over 12 attack families.
2. **Autoencoder** — trained on BENIGN only; reconstruction error spikes for anything unusual.

| MLP says | AE says | Final |
|---|---|---|
| Attack X (high conf) | normal | Attack X |
| Attack X (high conf) | anomaly | Attack X (consistent) |
| BENIGN | normal | BENIGN |
| BENIGN | **anomaly** | **UNKNOWN_ANOMALY** (likely zero-day) |
| any (low conf) | anomaly | UNKNOWN_ANOMALY |

The `confidence_threshold` (default `0.4`) and AE behaviour can be tuned at
construction time:

```python
predictor = SmartTIDS_Predictor(confidence_threshold=0.6, use_autoencoder=False)
```

---

## 7. Feature contract

The model expects the 77 CICIDS2017 flow features listed in
`data/cicids2017/features/feature_names.json`. You can fetch the exact list at
runtime:

```python
predictor.expected_features()
```

Missing features are silently filled with `0.0`; extra features are dropped.
This makes the API tolerant to small schema drift in the upstream flow exporter.

---

## 8. Re-training

If new data arrives or you want to retune:

1. Drop new CSVs into `data/cicids2017/raw/`.
2. Re-run notebooks `01` and `02` (regenerates splits + balanced set + scaler).
3. Re-run notebooks `03` and `04` (overwrites `models/*.keras`).
4. Bump `MODEL_VERSION` in `src/config.py`.

The inference module needs no code changes — it always loads from
`models/` at construction time.

---

## 9. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError: scaler.joblib` | Notebook 01 was not run |
| `FileNotFoundError: mlp_model.keras` | Notebook 03 was not run |
| AE warning "threshold file missing" | Notebook 04 was not run; predictor falls back to a permissive default |
| "Array input has N features, expected 77" | Pass a `dict` instead, or align column order to `predictor.expected_features()` |
| `ImportError: ctgan` | Only required for notebook 02 (training); not needed at inference |

---

## 10. Contact

PFA project — AI-SmartTIDS. AI module owner: ML team.
For integration questions, raise an issue against the AI module repository
or reach the maintainer at `7layemm@gmail.com`.
