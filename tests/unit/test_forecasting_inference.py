"""Unit tests for SageMaker forecasting inference handlers."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


@pytest.fixture
def fcst(modules):
    return modules["forecast_inference"]


@pytest.fixture
def mock_model_dir(tmp_path: Path) -> str:
    """Train a trivial LinearRegression to stand in for XGBoost — picklable."""
    feature_cols = ["aqi_lag_1h", "aqi_lag_24h", "hour_sin", "city_code"]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = X @ np.array([1.0, 0.5, 2.0, 0.1]) + 80

    model = LinearRegression().fit(X, y)
    joblib.dump(model, tmp_path / "xgb_model.joblib")

    (tmp_path / "feature_columns.json").write_text(json.dumps(feature_cols))
    (tmp_path / "city_mapping.json").write_text(
        json.dumps({"0": "ha-noi", "1": "ho-chi-minh-city"})
    )
    return str(tmp_path)


class TestModelFn:
    def test_returns_bundle_with_required_keys(self, fcst, mock_model_dir):
        bundle = fcst.model_fn(mock_model_dir)
        assert set(bundle) == {"model", "features", "city_to_code"}
        assert bundle["features"] == ["aqi_lag_1h", "aqi_lag_24h", "hour_sin", "city_code"]
        assert bundle["city_to_code"] == {"ha-noi": 0, "ho-chi-minh-city": 1}


class TestInputFn:
    def test_parses_csv(self, fcst):
        df = fcst.input_fn("aqi_lag_1h,aqi_lag_24h\n80,90\n75,95", "text/csv")
        assert list(df.columns) == ["aqi_lag_1h", "aqi_lag_24h"]
        assert len(df) == 2

    def test_parses_bytes(self, fcst):
        df = fcst.input_fn(b"aqi,pm25\n80,60", "text/csv")
        assert df.iloc[0]["aqi"] == 80

    def test_parses_json_lines(self, fcst):
        body = '{"aqi": 80, "pm25": 60}\n{"aqi": 75, "pm25": 55}'
        df = fcst.input_fn(body, "application/json")
        assert len(df) == 2
        assert df.iloc[1]["aqi"] == 75


class TestPredictFn:
    def test_maps_city_to_code_when_missing(self, fcst, mock_model_dir):
        bundle = fcst.model_fn(mock_model_dir)
        df = pd.DataFrame({
            "queried_city": ["ha-noi", "ho-chi-minh-city"],
            "aqi_lag_1h": [80, 60],
            "aqi_lag_24h": [90, 55],
            "hour_sin": [0.5, -0.3],
        })
        preds = fcst.predict_fn(df, bundle)
        assert preds.shape == (2,)

    def test_raises_on_missing_features(self, fcst, mock_model_dir):
        bundle = fcst.model_fn(mock_model_dir)
        df = pd.DataFrame({"aqi_lag_1h": [80]})
        with pytest.raises(ValueError, match="Missing input features"):
            fcst.predict_fn(df, bundle)

    def test_unknown_city_coded_as_minus_one(self, fcst, mock_model_dir):
        bundle = fcst.model_fn(mock_model_dir)
        df = pd.DataFrame({
            "queried_city": ["unknown-city"],
            "aqi_lag_1h": [80],
            "aqi_lag_24h": [90],
            "hour_sin": [0.5],
        })
        fcst.predict_fn(df, bundle)
        assert (df["city_code"] == -1).all()


class TestOutputFn:
    def test_csv_output(self, fcst):
        out = fcst.output_fn(np.array([95.0, 102.5]), "text/csv")
        assert out.startswith("predicted_aqi\n")
        assert "95.0" in out

    def test_json_output(self, fcst):
        out = fcst.output_fn(np.array([95.0, 102.5]), "application/json")
        lines = [json.loads(line) for line in out.strip().splitlines()]
        assert lines[0] == {"predicted_aqi": 95.0}
