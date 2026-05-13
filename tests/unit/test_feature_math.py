"""Test the math of feature engineering.

Since the Glue feature job runs on PySpark (expensive to spin up in unit
tests), we verify the underlying formulas with pandas/numpy. If these hold,
the equivalent Spark expressions in `glue_jobs/silver_to_ml_features.py`
should produce the same values.
"""
import numpy as np
import pandas as pd
import pytest


def _cyclical(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    theta = 2 * np.pi * values / period
    return np.sin(theta), np.cos(theta)


class TestCyclicalEncoding:
    def test_hour_sin_cos_periodic(self):
        hours = np.array([0, 6, 12, 18])
        h_sin, h_cos = _cyclical(hours, 24.0)
        # 0h vs 24h should be identical (period closure)
        assert np.isclose(h_sin[0], np.sin(0))
        assert np.isclose(h_cos[0], 1.0)
        assert np.isclose(h_sin[2], np.sin(np.pi))
        assert np.isclose(h_cos[2], -1.0)

    def test_month_cyclical_is_continuous(self):
        """Dec (12) and Jan (1) should be close in (sin, cos) space."""
        s12, c12 = _cyclical(np.array([12]), 12.0)
        s1, c1 = _cyclical(np.array([1]), 12.0)
        dist = np.sqrt((s12 - s1) ** 2 + (c12 - c1) ** 2)
        # Linear encoding would give |12 - 1| = 11; cyclical gives a small distance
        assert dist[0] < 1.0

    def test_hour_encoding_bounded(self):
        hours = np.arange(24)
        h_sin, h_cos = _cyclical(hours, 24.0)
        assert np.all((h_sin >= -1) & (h_sin <= 1))
        assert np.all((h_cos >= -1) & (h_cos <= 1))


class TestLagFeatureMath:
    def test_lag_1h_shifts_by_one(self, sample_aqi_df):
        df = sample_aqi_df.sort_values(["queried_city", "measured_at"]).copy()
        df["aqi_lag_1h"] = df.groupby("queried_city")["aqi"].shift(1)

        # First row per city has NaN lag
        firsts = df.groupby("queried_city").head(1)
        assert firsts["aqi_lag_1h"].isna().all()

        # Second row's lag == first row's aqi (per city)
        for city in df["queried_city"].unique():
            sub = df[df["queried_city"] == city].reset_index(drop=True)
            assert sub.loc[1, "aqi_lag_1h"] == pytest.approx(sub.loc[0, "aqi"])

    def test_rolling_mean_24h(self, sample_aqi_df):
        df = sample_aqi_df.sort_values(["queried_city", "measured_at"]).copy()
        df["rolling_24h"] = (
            df.groupby("queried_city")["aqi"]
              .transform(lambda s: s.rolling(24, min_periods=1).mean())
        )
        # Rolling mean should be within [min, max] of the window
        assert df["rolling_24h"].min() >= sample_aqi_df["aqi"].min() - 1e-9
        assert df["rolling_24h"].max() <= sample_aqi_df["aqi"].max() + 1e-9


class TestTargetCreation:
    def test_target_is_shifted_forward_24h(self, sample_aqi_df):
        """aqi_target(t) = aqi(t + 24h) — forecasting target."""
        df = sample_aqi_df.sort_values(["queried_city", "measured_at"]).copy()
        df["aqi_target"] = df.groupby("queried_city")["aqi"].shift(-24)

        # Last 24h per city must be NaN (no future data)
        last_24 = df.groupby("queried_city").tail(24)
        assert last_24["aqi_target"].isna().all()

        # Row at hour=0 should have target == aqi at hour=24 in same city
        city = df["queried_city"].iloc[0]
        sub = df[df["queried_city"] == city].reset_index(drop=True)
        assert sub.loc[0, "aqi_target"] == pytest.approx(sub.loc[24, "aqi"])
