"""
ML utilities: feature engineering, XGBoost training, SHAP, Isolation Forest.

Uses daily aggregation (most stations report 1-4 readings/day), so all
lag/rolling features are in days rather than hours.

All expensive computations use @st.cache_resource — runs once per server session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Daily-resolution feature set
FORECAST_FEATURES = [
    "aqi_lag_1d", "aqi_lag_2d", "aqi_lag_3d", "aqi_lag_5d", "aqi_lag_7d",
    "aqi_roll_mean_3d", "aqi_roll_std_3d",
    "aqi_roll_mean_7d", "aqi_roll_std_7d",
    "aqi_roll_mean_14d",
    "aqi_diff_1d",
    "pm25", "pm10", "temperature", "humidity", "pressure", "wind",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_dry_season",
]

ANOMALY_FEATURES = ["aqi", "pm25", "pm10", "temperature", "humidity", "pressure", "wind"]

SHAP_FEATURE_LABELS = {
    "aqi_lag_1d": "AQI lag 1 day",
    "aqi_lag_2d": "AQI lag 2 days",
    "aqi_lag_3d": "AQI lag 3 days",
    "aqi_lag_5d": "AQI lag 5 days",
    "aqi_lag_7d": "AQI lag 7 days",
    "aqi_roll_mean_3d": "Rolling avg 3d",
    "aqi_roll_std_3d": "Rolling std 3d",
    "aqi_roll_mean_7d": "Rolling avg 7d",
    "aqi_roll_std_7d": "Rolling std 7d",
    "aqi_roll_mean_14d": "Rolling avg 14d",
    "aqi_diff_1d": "AQI change 1d",
    "pm25": "PM2.5",
    "pm10": "PM10",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "wind": "Wind speed",
    "month_sin": "Month (sin)",
    "month_cos": "Month (cos)",
    "dow_sin": "Day-of-week (sin)",
    "dow_cos": "Day-of-week (cos)",
    "is_weekend": "Is weekend",
    "is_dry_season": "Dry season",
}


def _build_features_for_city(city_df: pd.DataFrame) -> pd.DataFrame:
    df = city_df.copy().sort_values("measured_at")
    # Daily aggregation — robust to irregular sampling
    df = df.set_index("measured_at").resample("D").mean(numeric_only=True)
    df = df.dropna(subset=["aqi"])  # keep only days with actual readings
    df = df.reset_index()

    aqi = df["aqi"]
    for lag in [1, 2, 3, 5, 7]:
        df[f"aqi_lag_{lag}d"] = aqi.shift(lag)

    for w, name in [(3, "3d"), (7, "7d"), (14, "14d")]:
        rolled = aqi.shift(1).rolling(w, min_periods=max(1, w // 2))
        df[f"aqi_roll_mean_{name}"] = rolled.mean()
        df[f"aqi_roll_std_{name}"] = rolled.std().fillna(0)

    df["aqi_diff_1d"] = aqi.diff(1)

    month = df["measured_at"].dt.month
    dow = df["measured_at"].dt.dayofweek
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_dry_season"] = month.isin([11, 12, 1, 2, 3, 4]).astype(int)
    df["aqi_target"] = aqi.shift(-1)  # next-day AQI

    return df


@st.cache_data(show_spinner=False)
def build_features(_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for city, grp in _df.groupby("queried_city"):
        feat = _build_features_for_city(grp)
        feat["queried_city"] = city
        frames.append(feat)
    return pd.concat(frames, ignore_index=True)


@st.cache_resource(show_spinner="Training forecast model…")
def get_forecast_bundle():
    """
    Train XGBoost next-day AQI forecast and compute SHAP values.
    Returns metrics, test predictions, SHAP importance, and raw SHAP values.
    """
    import shap
    from xgboost import XGBRegressor

    from dashboard.utils.data_loader import load_raw

    raw = load_raw()
    features_df = build_features(raw)

    available = [f for f in FORECAST_FEATURES if f in features_df.columns]
    # Only require lag_1d and target to be non-null; others filled/optional
    req_cols = ["aqi_lag_1d", "aqi_target"]
    subset = features_df.dropna(subset=req_cols).copy()
    # Forward-fill sparse weather readings, then fill remaining NaN with median
    for col in available:
        if col not in req_cols:
            subset[col] = subset[col].ffill().bfill().fillna(subset[col].median())

    X = subset[available].values
    y = subset["aqi_target"].values
    dates = pd.to_datetime(subset["measured_at"].values)
    cities = subset["queried_city"].values

    split_ts = pd.Timestamp("2021-09-01")
    train_mask = dates < split_ts

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    dates_test = dates[~train_mask]
    cities_test = cities[~train_mask]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    mape = float(np.mean(np.abs((y_pred - y_test) / (y_test + 1e-6))) * 100)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_sample_n = min(400, len(X_test))
    X_shap = X_test[:shap_sample_n]
    shap_values = explainer.shap_values(X_shap)

    shap_importance = pd.DataFrame({
        "feature": available,
        "label": [SHAP_FEATURE_LABELS.get(f, f) for f in available],
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    test_df = pd.DataFrame({
        "measured_at": dates_test,
        "queried_city": cities_test,
        "actual": y_test,
        "predicted": y_pred,
        "error": y_pred - y_test,
    })

    return {
        "model": model,
        "explainer": explainer,
        "shap_values": shap_values,
        "X_shap": X_shap,
        "feature_names": available,
        "shap_importance": shap_importance,
        "test_df": test_df,
        "metrics": {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape},
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
    }


@st.cache_resource(show_spinner="Running anomaly detection…")
def get_anomaly_bundle():
    """Isolation Forest on daily aggregated data. Returns df with anomaly flags."""
    from dashboard.utils.data_loader import get_daily_summary

    daily = get_daily_summary()
    # daily uses avg_ prefix for all numeric columns
    candidate_cols = ["avg_aqi", "avg_pm25", "avg_pm10",
                      "avg_temperature", "avg_humidity", "avg_pressure", "avg_wind"]
    avail = [c for c in candidate_cols if c in daily.columns]

    df_clean = daily.dropna(subset=avail).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[avail])

    iforest = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
    df_clean["is_anomaly"] = iforest.fit_predict(X_scaled) == -1
    df_clean["anomaly_score"] = -iforest.score_samples(X_scaled)

    # Alias columns for display compatibility
    df_clean["aqi"] = df_clean["avg_aqi"]
    if "avg_pm25" in df_clean.columns:
        df_clean["pm25"] = df_clean["avg_pm25"]
    if "avg_pm10" in df_clean.columns:
        df_clean["pm10"] = df_clean["avg_pm10"]
    if "avg_temperature" in df_clean.columns:
        df_clean["temperature"] = df_clean["avg_temperature"]
    if "avg_humidity" in df_clean.columns:
        df_clean["humidity"] = df_clean["avg_humidity"]
    df_clean["measured_at"] = df_clean["date"]

    return df_clean
