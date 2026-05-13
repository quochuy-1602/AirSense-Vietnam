"""
SageMaker Inference Handler — AQI Anomaly Detection (Isolation Forest)
───────────────────────────────────────────────────────────────────────
Used by SageMaker Batch Transform / Endpoint when deploying a model saved
by `sagemaker/anomaly/train.py`.

Input: CSV with header matching feature_columns.json (e.g. aqi, pm25, pm10, ...)
Output: CSV with columns [anomaly_score, is_anomaly]
"""

import io
import json
import os

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir: str):
    model = joblib.load(os.path.join(model_dir, "iforest_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json")) as f:
        features = json.load(f)
    with open(os.path.join(model_dir, "metadata.json")) as f:
        meta = json.load(f)
    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "threshold": float(meta.get("anomaly_score_threshold", 0.0)),
    }


def input_fn(request_body: bytes | str, content_type: str = "text/csv") -> pd.DataFrame:
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    if "json" in (content_type or "").lower():
        return pd.DataFrame(
            [json.loads(line) for line in request_body.strip().splitlines() if line]
        )
    return pd.read_csv(io.StringIO(request_body))


def predict_fn(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["features"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = scaler.transform(df[feature_cols].astype(float))
    scores = -model.score_samples(X)
    is_anomaly = (model.predict(X) == -1).astype(int)

    return pd.DataFrame({"anomaly_score": scores, "is_anomaly": is_anomaly})


def output_fn(df: pd.DataFrame, accept: str = "text/csv") -> str:
    if "json" in (accept or "").lower():
        return df.to_json(orient="records", lines=True)
    return df.to_csv(index=False)
