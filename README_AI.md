# AI-SmartTIDS — AI Module

Real-time intelligent **Intrusion Detection System** built on the
CICIDS2017 dataset. This document is the integration guide for the
**cyber-security team**: how to install, run, and call the AI module,
plus everything the ML team did so the system is reproducible.

---

## 1. What this module does

Given a network **flow** (77 numeric statistics — duration, packet counts,
byte rates, IAT stats, TCP flag counts, etc.), the AI module returns:

* whether the flow is benign or one of **11 attack families**
  (DDoS, DoS Hulk, DoS GoldenEye, DoS Slowhttptest, DoS slowloris,
  PortScan, FTP-Patator, SSH-Patator, Bot, Web Attack Brute Force,
  Web Attack XSS),
* a confidence score and the top-3 candidate classes,
* an `is_anomaly` flag from an autoencoder backstop (catches flows that
  don't look like normal traffic — useful as a zero-day net),
* a recommended **severity** + **action** for the SOC.

The system is a **hybrid of two models**:

| Model       | Purpose                          | Strength                       |
| ----------- | -------------------------------- | ------------------------------ |
| MLP (Keras) | Multi-class classifier           | Signature attacks (DDoS, Patators, PortScan, Web Attacks) |
| Autoencoder | Unsupervised anomaly detector    | DoS-family + unknown anomalies (zero-days) |

The two are reconciled inside `SmartTIDS_Predictor` (see §6).

---

## 2. Project layout

```
smart_ids/
├── data/cicids2017/
│   ├── raw/                       # original CICIDS2017 CSVs (you provide)
│   ├── processed/                 # train/val/test (built by notebook 01)
│   └── features/                  # label_map.json, feature_names.json
├── models/                        # trained artifacts (built by notebooks 02 & 03)
│   ├── scaler.joblib
│   ├── mlp_model.keras
│   ├── autoencoder_model.keras
│   └── autoencoder_threshold.json
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_mlp_training.ipynb
│   ├── 03_autoencoder_training.ipynb
│   └── 04_evaluation_and_comparison.ipynb
├── src/
│   ├── config.py                  # paths, hyperparams, attack policy
│   ├── preprocessing.py           # cleaning + label normalisation + feature alignment
│   ├── inference.py               # ★ SmartTIDS_Predictor — Python entry point
│   ├── api.py                     # ★ FastAPI service (REST entry point)
│   └── flow_extractor.py          # ★ pcap -> 77-feature dicts
├── tests/
│   └── test_inference.py          # 14 unit tests
├── requirements.txt
└── README_AI.md
```

The three files marked ★ are **everything the cyber team needs**.

---

## 3. Installation

```powershell
# from the project root (Windows PowerShell)
python -m venv ids_pfa
.\ids_pfa\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU optional but recommended for training (CPU works for inference at
~6,000 flows/sec on a laptop).

---

## 4. Training pipeline (one-time setup)

Only required if you don't already have `models/*.keras`. From the cyber
team's perspective this is "build the artifacts the API loads". Run the
notebooks in order:

```bash
jupyter notebook notebooks/01_eda_and_preprocessing.ipynb     # cleans, splits (stratified), fits scaler
jupyter notebook notebooks/02_mlp_training.ipynb              # trains MLP w/ class-weights
jupyter notebook notebooks/03_autoencoder_training.ipynb      # trains AE on BENIGN
jupyter notebook notebooks/04_evaluation_and_comparison.ipynb # eval, SHAP, latency
```

After 02 + 03 complete, the `models/` directory has everything inference
needs. Imbalance is handled in the **loss function** (sklearn-balanced
class weights, capped at 10x) — no GAN, no resampling. Notebooks 02
and 03 also include a **hyperparameter search** (random search +
3-fold stratified CV on a subsample) whose best configurations are
persisted as `models/mlp_best_params.json` and `models/ae_best_params.json`.

> **Note for the cyber team:** the trained `models/` directory (~3 MB) is
> shipped with the repo, so you can skip the training notebooks entirely
> and use the predictor / API immediately after `pip install`.

---

## 5. Three ways to call the AI module

Pick whichever fits your stack.

### 5.A — Python import (in-process)

Best for Python services that already live alongside the AI module.

```python
from src.inference import SmartTIDS_Predictor

predictor = SmartTIDS_Predictor()      # loads scaler + MLP + AE + label map

flow = {
    "Flow Duration":           5_120_343,
    "Total Fwd Packets":       12,
    "Total Backward Packets":  10,
    "Flow Bytes/s":            1023.4,
    # ... any subset of the 77 CICIDS features (missing ones default to 0)
}

result = predictor.predict_flow(flow)
print(result)
```

### 5.B — REST API (recommended for cross-language integration)

This is the right entry point for **anything that's not Python**: SIEM
sidecars, Suricata extensions, custom dashboards, Java/Go/Node services.

```powershell
# start the service
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

| Endpoint               | Method | Purpose                                           |
| ---------------------- | ------ | ------------------------------------------------- |
| `/health`              | GET    | Liveness + model metadata. Use for readiness probe |
| `/features`            | GET    | The 77 feature names the model expects            |
| `/model/info`          | GET    | Class catalogue + per-class severity/action policy |
| `/predict`             | POST   | Classify a single flow                            |
| `/predict/batch`       | POST   | Classify many flows in one call                   |

Example call (PowerShell):

```powershell
$body = @{ "Flow Duration" = 5120343; "Total Fwd Packets" = 12 } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri http://localhost:8000/predict `
                  -ContentType "application/json" -Body $body
```

Example call (curl):

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"Flow Duration": 5120343, "Total Fwd Packets": 12}'
```

### 5.C — Pcap → predictions (offline analysis & demo)

For analysing captured traffic end-to-end. Two backends (full-fidelity
CICFlowMeter or pure-Python scapy fallback) live in
`src/flow_extractor.py`.

CLI:

```bash
# extract features only
python -m src.flow_extractor capture.pcap --json flows.json

# extract + classify in one shot
python -m src.flow_extractor capture.pcap --predict --json predictions.json
```

Programmatic:

```python
from src.flow_extractor import extract_with_scapy
from src.inference import SmartTIDS_Predictor

flows  = extract_with_scapy("capture.pcap")
results = SmartTIDS_Predictor().predict_batch(flows)
```

For production accuracy use the official **CICFlowMeter** (Java) tool,
which produces the exact 77-feature schema CICIDS was labelled with.
Helper provided:

```python
from src.flow_extractor import extract_with_cicflowmeter
csv_path = extract_with_cicflowmeter("capture.pcap", out_dir="flows/")
# then read with pandas and feed predictor.predict_batch(...)
```

---

## 6. Response schema

Every prediction returns the same JSON shape:

```json
{
  "is_anomaly": true,
  "anomaly_score": 0.0823,
  "anomaly_threshold": 0.0333,
  "predicted_class": "DDoS",
  "predicted_class_id": 2,
  "confidence": 0.9731,
  "top_3": [
    {"class": "DDoS",     "probability": 0.9731},
    {"class": "DoS Hulk", "probability": 0.0211},
    {"class": "PortScan", "probability": 0.0035}
  ],
  "severity": "CRITICAL",
  "recommended_action": "BLOCK_IMMEDIATELY",
  "inference_time_ms": 4.2,
  "model_version": "1.0.0"
}
```

| Field                | Type    | Meaning                                                   |
| -------------------- | ------- | --------------------------------------------------------- |
| `is_anomaly`         | bool    | AE reconstruction error exceeded the saved threshold       |
| `anomaly_score`      | float   | Reconstruction MSE for this flow                          |
| `anomaly_threshold`  | float   | Threshold the AE was tuned to                             |
| `predicted_class`    | str     | Attack family or `BENIGN` or `UNKNOWN_ANOMALY`            |
| `predicted_class_id` | int     | Class index, `-1` for `UNKNOWN_ANOMALY`                   |
| `confidence`         | float   | MLP top-1 probability                                     |
| `top_3`              | list    | Top-3 candidate classes with probabilities                |
| `severity`           | str     | `INFO` / `MEDIUM` / `HIGH` / `CRITICAL` / `UNKNOWN`       |
| `recommended_action` | str     | `ALLOW`, `MONITOR`, `ALERT_AND_MONITOR`, `RATE_LIMIT_AND_ALERT`, `BLOCK_SOURCE_IP`, `BLOCK_AND_INVESTIGATE`, `ALERT_AND_INVESTIGATE`, `BLOCK_IMMEDIATELY` |
| `inference_time_ms`  | float   | Wall-clock latency for this prediction                    |
| `model_version`      | str     | Tied to artifacts under `models/`                         |

### Hybrid decision logic

| MLP says               | AE says    | Final label           | Why                                  |
| ---------------------- | ---------- | --------------------- | ------------------------------------ |
| Attack X (high conf)   | normal     | Attack X              | Both agree it's not benign           |
| Attack X (high conf)   | anomaly    | Attack X (consistent) | AE confirms it's unusual             |
| BENIGN (high conf)     | normal     | BENIGN                | Both agree it's benign               |
| BENIGN                 | **anomaly**| **UNKNOWN_ANOMALY**   | Likely zero-day; not in training set |
| any (low conf < 0.4)   | anomaly    | UNKNOWN_ANOMALY       | Model unsure + AE flagged            |

---

## 7. Performance

Measured on a laptop CPU (no GPU), test set = 424,172 held-out flows:

| Metric                                | Value             |
| ------------------------------------- | ----------------- |
| Single-flow latency (mean)            | ~0.56 ms          |
| Batch throughput                      | ~1,800 flows/s    |
| MLP weighted-F1 (test set)            | ~0.98             |
| MLP macro-F1 (test set)               | 0.7506            |
| AE ROC-AUC (binary attack/benign)     | 0.8453            |
| Hybrid binary recall (attack found)   | 0.9996            |
| AE recall on DoS variants             | 60 – 95 %         |

Numbers are reproducible by running notebook 04.

---

## 8. Testing

```bash
pytest tests/ -v
```

Tests cover artifact loading, every input shape (dict / list / numpy /
DataFrame), missing/extra-feature tolerance, batch ↔ single agreement,
JSON-serialisability, and the response schema. They auto-skip if the
trained artifacts aren't present yet.

---

## 9. Re-training

If new data arrives or you want to retune:

1. Drop new CSVs into `data/cicids2017/raw/`.
2. Re-run notebook **01** (regenerates splits + scaler).
3. Re-run notebooks **02** and **03** (overwrites `models/*.keras`).
4. Bump `MODEL_VERSION` in `src/config.py`.

The inference / API / extractor layers need **no code change** — they
always load from `models/` at startup.

### Web Attack label normalisation (already handled)

The raw CICIDS2017 CSVs contain a latin-1 `\x96` byte between
"Web Attack" and the attack name (e.g. `Web Attack \x96 Brute Force`),
which renders as garbage in JSON. This is now handled automatically by
`src.preprocessing.normalize_labels()` — called both in notebook 01
and inside the production preprocessing path — so labels are always
the clean form `Web Attack - Brute Force` / `Web Attack - XSS`. The
keys in `src/config.py:ATTACK_POLICY` match that clean form.

No manual fix needed when re-training.

---

## 10. Honest caveats (what to flag in the SOC handover)

* **Web Attack XSS / Bot / Web Attack Brute Force** have very few training
  samples (98 – 294 in test). F1 on these classes is low. They're
  data-limited, not model-limited; closing the gap requires more labelled
  data of these specific attacks.
* The autoencoder is **good at DoS** (60 – 95 % recall) but **poor at
  PortScan / brute-force / Web Attacks** because those look statistically
  identical to benign traffic at the single-flow level. The MLP handles
  those classes; the AE backs up DoS detection and serves as the zero-day
  net.
* The pure-Python `extract_with_scapy` is a fallback. For production
  traffic, install **CICFlowMeter** to get the full 77-feature fidelity
  the dataset was labelled with.

---

## 11. Cyber-team checklist

A minimal "is the AI module ready to integrate?" checklist:

- [ ] `pip install -r requirements.txt` succeeds
- [ ] `models/` contains `scaler.joblib`, `mlp_model.keras`,
      `autoencoder_model.keras`, `autoencoder_threshold.json`
- [ ] `python -m src.inference --healthcheck` prints `"status": "ok"`
- [ ] `pytest tests/` is all green
- [ ] `uvicorn src.api:app --port 8000` starts and `/health` returns 200
- [ ] `curl -X POST localhost:8000/predict -d '{}' -H "Content-Type: application/json"`
      returns a valid JSON prediction (the empty-dict case — every
      missing feature defaults to 0)

---

## 12. Troubleshooting

| Symptom | Cause / Fix |
| ------- | ----------- |
| `FileNotFoundError: scaler.joblib` | Notebook 01 was not run |
| `FileNotFoundError: mlp_model.keras` | Notebook 02 was not run |
| API logs `predictor not loaded` | Lifespan startup failed — check the prior log line for the real error |
| MLP only predicts BENIGN | `MLP_CONFIG["imbalance_strategy"]` is `"none"` — switch to `"class_weights"` or `"focal"` in `src/config.py` and retrain |
| `ImportError: scapy` | Only required for `flow_extractor.py`. `pip install scapy` |
| `ImportError: shap` | Only required for notebook 04 SHAP cell |

---

## 13. Contact

PFA project — **AI-SmartTIDS**.
AI module owner: ML team. For integration questions, open an issue on
the repository or reach the maintainer at `7layemm@gmail.com`.
