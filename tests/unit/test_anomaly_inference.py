"""Unit tests for SageMaker anomaly inference handlers."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def anom(modules):
    return modules["anomaly_inference"]


@pytest.fixture
def mock_model_dir(tmp_path: Path) -> str:
    """Train a trivial Isolation Forest + scaler on 2 features."""
    features = ["aqi", "pm25"]
    rng = np.random.default_rng(0)
    X_train = rng.normal(50, 10, size=(100, 2))

    scaler = StandardScaler().fit(X_train)
    model = IsolationForest(n_estimators=20, contamination=0.05, random_state=0).fit(
        scaler.transform(X_train)
    )

    joblib.dump(model, tmp_path / "iforest_model.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    (tmp_path / "feature_columns.json").write_text(json.dumps(features))
    (tmp_path / "metadata.json").write_text(json.dumps({"anomaly_score_threshold": 0.6}))
    return str(tmp_path)


def test_model_fn_loads_full_bundle(anom, mock_model_dir):
    bundle = anom.model_fn(mock_model_dir)
    assert set(bundle) == {"model", "scaler", "features", "threshold"}
    assert bundle["threshold"] == pytest.approx(0.6)


def test_predict_fn_flags_obvious_outlier(anom, mock_model_dir):
    bundle = anom.model_fn(mock_model_dir)
    df = pd.DataFrame({
        "aqi":  [50.0, 500.0],   # 2nd row is a clear outlier
        "pm25": [40.0, 450.0],
    })
    result = anom.predict_fn(df, bundle)
    assert list(result.columns) == ["anomaly_score", "is_anomaly"]
    assert len(result) == 2
    # Outlier score should be higher than normal score
    assert result.iloc[1]["anomaly_score"] > result.iloc[0]["anomaly_score"]


def test_predict_fn_raises_on_missing(anom, mock_model_dir):
    bundle = anom.model_fn(mock_model_dir)
    with pytest.raises(ValueError, match="Missing features"):
        anom.predict_fn(pd.DataFrame({"aqi": [50]}), bundle)


def test_output_fn_roundtrip(anom):
    df = pd.DataFrame({"anomaly_score": [0.42, 0.89], "is_anomaly": [0, 1]})
    csv_out = anom.output_fn(df, "text/csv")
    json_out = anom.output_fn(df, "application/json")
    assert "anomaly_score" in csv_out
    records = [json.loads(line) for line in json_out.strip().splitlines()]
    assert records[1] == {"anomaly_score": 0.89, "is_anomaly": 1}
