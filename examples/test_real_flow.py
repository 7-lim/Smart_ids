"""
Quick API smoke test — send real CICIDS rows through /predict and compare
the prediction against the ground-truth label.

Run after starting the API:
    uvicorn src.api:app --host 127.0.0.1 --port 8000

Usage:
    python examples/test_real_flow.py
    python examples/test_real_flow.py --rows 5
    python examples/test_real_flow.py --url http://other:9000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from urllib.error import URLError

import pandas as pd


def normalize(s: str) -> str:
    """API returns clean labels; raw CSVs have a latin-1 byte. Make them comparable."""
    return s.replace("\x96", "-")


def http_post(url: str, payload: dict, timeout: float = 10.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="data/cicids2017/processed/test.csv")
    ap.add_argument("--url",  default="http://127.0.0.1:8000")
    ap.add_argument("--rows", type=int, default=2, help="Rows per attack class")
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        sys.exit(f"Test CSV not found: {csv}. Run notebook 01 first.")

    # 1) Confirm the API is up.
    try:
        h = http_get(args.url + "/health")
        print(f"API ready: model={h['model_version']}, classes={h['n_classes']}, "
              f"AE={'on' if h['autoencoder_loaded'] else 'off'}\n")
    except URLError as e:
        sys.exit(f"Cannot reach {args.url}: {e}\n"
                 f"Start it first:  uvicorn src.api:app --port 8000")

    # 2) Sample rows per class (tolerant when a class has fewer than --rows).
    test = pd.read_csv(csv)
    sample = (test.groupby("Label", group_keys=False)
                  .apply(lambda g: g.sample(n=min(args.rows, len(g)), random_state=0)))

    # 3) Send each one to /predict and compare.
    print(f"{'TRUE':<28} {'PREDICTED':<28} {'CONF':>6}  {'SEV':<10}  ok")
    print("-" * 90)

    correct = 0
    for _, row in sample.iterrows():
        true_label = normalize(row["Label"])
        flow = row.drop("Label").to_dict()
        try:
            r = http_post(args.url + "/predict", flow)
        except Exception as e:
            print(f"  ERROR on {true_label}: {e}")
            continue

        pred = r["predicted_class"]
        ok = pred == true_label
        if ok:
            correct += 1
        mark = "OK" if ok else "  "
        print(f"{true_label:<28} {pred:<28} {r['confidence']:6.3f}  {r['severity']:<10}  {mark}")

    total = len(sample)
    print("-" * 90)
    print(f"Matched {correct}/{total}  ({correct/total:.1%})")


if __name__ == "__main__":
    main()
