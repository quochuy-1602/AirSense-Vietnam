"""Shared pytest fixtures.

Makes the Lambda / SageMaker / Glue source paths importable during tests,
and provides common fixtures (moto-backed AWS, sample DataFrames).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(alias: str, file_path: Path):
    """Load a file as a uniquely-aliased module (avoids name collisions).

    Needed because the repo has several `lambda_function.py` and `inference.py`
    files — each must be addressable by a distinct name at test time.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(alias, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {alias} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# Pre-load all source modules with distinct aliases
_MODULES_TO_LOAD = {
    "ingestion_lambda":      PROJECT_ROOT / "lambdas" / "air-quality-api-ingestion" / "lambda_function.py",
    "dq_check_lambda":       PROJECT_ROOT / "lambdas" / "quality_data" / "lambda_function.py",
    "ml_inference_lambda":   PROJECT_ROOT / "lambdas" / "ml_inference" / "lambda_function.py",
    "forecast_inference":    PROJECT_ROOT / "sagemaker" / "forecasting" / "inference.py",
    "anomaly_inference":     PROJECT_ROOT / "sagemaker" / "anomaly" / "inference.py",
}


@pytest.fixture(scope="session")
def modules():
    """Dict of loaded source modules — use this in tests instead of `import`."""
    # AWS credentials + region must be set before Lambda modules import boto3.
    os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

    os.environ.setdefault("ATHENA_DATABASE", "test_silver")
    os.environ.setdefault("ATHENA_OUTPUT_LOCATION", "s3://test/athena/")
    os.environ.setdefault("SNS_ALERT_TOPIC_ARN", "arn:aws:sns:ap-southeast-2:123456789012:test")
    os.environ.setdefault("ML_FEATURES_BUCKET", "test-ml-bucket")
    os.environ.setdefault("GOLD_BUCKET", "test-gold-bucket")
    os.environ.setdefault("GOLD_DATABASE", "test-gold-db")
    os.environ.setdefault("BATCH_OUTPUT_BUCKET", "test-ml-bucket")
    os.environ.setdefault("FORECAST_MODEL_PKG_GROUP", "aqi-forecast-models")
    os.environ.setdefault("ANOMALY_MODEL_PKG_GROUP", "aqi-anomaly-models")
    os.environ.setdefault("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/Test")
    os.environ.setdefault("WAQI_API_TOKEN", "test-token")
    os.environ.setdefault("S3_BUCKET_BRONZE", "test-bronze")

    loaded = {}
    for alias, path in _MODULES_TO_LOAD.items():
        if path.exists():
            loaded[alias] = _load_module(alias, path)
    return loaded


# ── Default AWS environment — keeps boto3/moto happy even without creds ─────
@pytest.fixture(autouse=True)
def _aws_test_env(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-southeast-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")


# ── Synthetic AQI time-series ────────────────────────────────────────────────
@pytest.fixture
def sample_aqi_df() -> pd.DataFrame:
    """Deterministic hourly AQI data for 2 cities × 7 days."""
    rng = np.random.default_rng(42)
    cities = ["ha-noi", "ho-chi-minh-city"]
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for city in cities:
        base = 80 if city == "ha-noi" else 55
        for hour in range(7 * 24):
            ts = start + timedelta(hours=hour)
            aqi = base + 20 * np.sin(hour * np.pi / 12) + rng.normal(0, 5)
            rows.append({
                "queried_city": city,
                "measured_at": ts,
                "aqi": float(max(0, aqi)),
                "pm25": float(max(0, aqi * 0.7)),
                "pm10": float(max(0, aqi * 0.9)),
                "humidity": 60.0 + rng.normal(0, 5),
                "temperature": 25.0 + rng.normal(0, 3),
                "pressure": 1013.0,
                "wind": 3.0,
                "source": "kaggle",
            })
    return pd.DataFrame(rows)
