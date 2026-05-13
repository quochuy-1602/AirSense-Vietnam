"""
Data loading utilities for the AirSense Vietnam dashboard.

Reads the raw Kaggle CSV and produces analysis-ready DataFrames.
All heavy I/O is cached with @st.cache_data so pages reload instantly.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = _ROOT / "raw_data_csv" / "historical_air_quality_2021_en.csv"

CITY_LABELS: dict[str, str] = {
    "ha-noi": "Ha Noi",
    "ho-chi-minh-city": "Ho Chi Minh City",
    "gia-lai": "Gia Lai",
    "quang-ninh": "Quang Ninh",
    "thua-thien-hue": "Thua Thien Hue",
    "bac-ninh": "Bac Ninh",
    "lao-cai": "Lao Cai",
    # kept for backward-compatibility with future data expansions
    "da-nang": "Da Nang",
    "cao-bang": "Cao Bang",
}

# (lat, lon) for map markers
CITY_COORDS: dict[str, tuple[float, float]] = {
    "ha-noi": (21.0285, 105.8542),
    "ho-chi-minh-city": (10.8231, 106.6297),
    "gia-lai": (13.9833, 108.0000),
    "quang-ninh": (21.0064, 107.2925),
    "thua-thien-hue": (16.4637, 107.5909),
    "bac-ninh": (21.1861, 106.0763),
    "lao-cai": (22.4856, 103.9754),
    "da-nang": (16.0544, 108.2022),
    "cao-bang": (22.6667, 106.2500),
}

AQI_LEVELS: list[tuple[int, int, str, str]] = [
    (0, 50, "Good", "#00E400"),
    (51, 100, "Moderate", "#FFFF00"),
    (101, 150, "Unhealthy for Sensitive", "#FF7E00"),
    (151, 200, "Unhealthy", "#FF0000"),
    (201, 300, "Very Unhealthy", "#8F3F97"),
    (301, 9999, "Hazardous", "#7E0023"),
]

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def get_aqi_level(aqi: float) -> tuple[str, str]:
    """Return (label, hex_color) for a given AQI value."""
    for lo, hi, label, color in AQI_LEVELS:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "#7E0023"


def _parse_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip().replace({"-": np.nan, "": np.nan}),
        errors="coerce",
    )


def _city_from_url(url: str) -> str:
    _MAP = {
        "hanoi": "ha-noi",
        "ho-chi-minh-city": "ho-chi-minh-city",
        "da-nang": "da-nang",
        "gia-lai": "gia-lai",
        "cao-bang": "cao-bang",
    }
    try:
        m = re.search(r"/city/vietnam/([^/]+)/", str(url))
        if m:
            return _MAP.get(m.group(1), m.group(1))
    except Exception:
        pass
    return "unknown"


@st.cache_data(show_spinner="Loading air quality data…")
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    df = df.rename(columns={
        "Station ID": "waqi_idx",
        "AQI index": "aqi",
        "Location": "location",
        "Station name": "station_name",
        "Url": "url",
        "Dominent pollutant": "dominant_pollutant",
        "CO": "co",
        "Dew": "dew",
        "Humidity": "humidity",
        "NO2": "no2",
        "O3": "o3",
        "Pressure": "pressure",
        "PM10": "pm10",
        "PM2.5": "pm25",
        "SO2": "so2",
        "Temperature": "temperature",
        "Wind": "wind",
        "Data Time S": "measured_at",
        "Data Time Tz": "tz",
        "Status": "status",
        "Alert level": "alert_level",
    })

    for col in ["aqi", "co", "humidity", "no2", "o3", "pressure", "pm10", "pm25",
                "so2", "temperature", "wind", "dew"]:
        if col in df.columns:
            df[col] = _parse_numeric(df[col])

    df["measured_at"] = pd.to_datetime(df["measured_at"], errors="coerce")
    df = df.dropna(subset=["measured_at", "aqi"])

    df["queried_city"] = df["url"].apply(_city_from_url)
    df = df[df["queried_city"] != "unknown"].copy()
    df["city_label"] = df["queried_city"].map(CITY_LABELS).fillna(df["queried_city"])

    df["hour"] = df["measured_at"].dt.hour
    df["month"] = df["measured_at"].dt.month
    df["dow"] = df["measured_at"].dt.dayofweek
    df["date"] = df["measured_at"].dt.normalize()

    return df.sort_values("measured_at").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_daily_summary() -> pd.DataFrame:
    df = load_raw()
    numeric = [c for c in ["aqi", "pm25", "pm10", "temperature", "humidity", "pressure", "wind"]
               if c in df.columns]

    daily = (
        df.groupby(["queried_city", "city_label", "date"])
        .agg(
            avg_aqi=("aqi", "mean"),
            max_aqi=("aqi", "max"),
            min_aqi=("aqi", "min"),
            **{f"avg_{c}": (c, "mean") for c in numeric if c != "aqi"},
        )
        .reset_index()
    )
    daily["pollution_level"] = daily["avg_aqi"].apply(lambda x: get_aqi_level(x)[0])
    daily["aqi_color"] = daily["avg_aqi"].apply(lambda x: get_aqi_level(x)[1])
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    return daily


@st.cache_data(show_spinner=False)
def get_monthly_summary() -> pd.DataFrame:
    daily = get_daily_summary()
    agg = dict(avg_aqi=("avg_aqi", "mean"), max_aqi=("max_aqi", "max"))
    if "avg_pm25" in daily.columns:
        agg["avg_pm25"] = ("avg_pm25", "mean")
    monthly = (
        daily.groupby(["queried_city", "city_label", "year", "month"])
        .agg(**agg)
        .reset_index()
    )
    monthly["aqi_rank"] = (
        monthly.groupby(["year", "month"])["avg_aqi"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    monthly["pollution_level"] = monthly["avg_aqi"].apply(lambda x: get_aqi_level(x)[0])
    return monthly
