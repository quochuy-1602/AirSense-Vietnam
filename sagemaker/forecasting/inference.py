"""
SageMaker Inference Handler — AQI Forecast (XGBoost)
─────────────────────────────────────────────────────
Used by SageMaker Batch Transform / Real-time Endpoint when deploying
an XGBoost model saved by `train.py`.

Expected input format (text/csv or application/json):
  - CSV with header; columns must match feature_columns.json
  - or JSON lines with {"feature": value, ...}

Output: CSV of predicted AQI (single column `predicted_aqi`).
"""

import io
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir: str):
    """Load model + feature columns."""
    model = joblib.load(os.path.join(model_dir, "xgb_model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(model_dir, "city_mapping.json")) as f:
        city_mapping = json.load(f)
    city_to_code = {v: int(k) for k, v in city_mapping.items()}
    return {"model": model, "features": feature_cols, "city_to_code": city_to_code}


def input_fn(request_body: bytes | str, content_type: str = "text/csv") -> pd.DataFrame:
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    ct = (content_type or "").lower()
    if "json" in ct:
        records = []
        for line in request_body.strip().splitlines():
            if line:
                records.append(json.loads(line))
        return pd.DataFrame(records)
    return pd.read_csv(io.StringIO(request_body))


def predict_fn(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    model = bundle["model"]
    feature_cols = bundle["features"]
    city_to_code = bundle["city_to_code"]

    if "queried_city" in df.columns and "city_code" not in df.columns:
        df["city_code"] = df["queried_city"].map(city_to_code).fillna(-1).astype(int)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing input features: {missing}")

    X = df[feature_cols]
    return model.predict(X)


def output_fn(preds: np.ndarray, accept: str = "text/csv") -> str:
    out = pd.DataFrame({"predicted_aqi": preds})
    if "json" in (accept or "").lower():
        return out.to_json(orient="records", lines=True)
    return out.to_csv(index=False)
