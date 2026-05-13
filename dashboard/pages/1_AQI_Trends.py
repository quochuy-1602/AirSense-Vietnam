"""
Page 1 — AQI Trends

Deep-dive into historical AQI patterns:
  • City-level time series with pollution level color bands
  • Daily min/max/avg range
  • Hourly average heatmap (city × hour)
  • Seasonal breakdown (dry vs wet season)
"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_loader import (
    AQI_LEVELS,
    CITY_LABELS,
    get_daily_summary,
    load_raw,
)

st.set_page_config(page_title="AQI Trends — AirSense", layout="wide")
st.title("📈 AQI Trends")

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
cities = list(CITY_LABELS.keys())
selected_cities = st.sidebar.multiselect(
    "Cities",
    options=cities,
    default=cities[:3],
    format_func=lambda c: CITY_LABELS[c],
)
if not selected_cities:
    st.warning("Select at least one city.")
    st.stop()

# ── Data ──────────────────────────────────────────────────────────────────────
raw = load_raw()
daily = get_daily_summary()

raw_f = raw[raw["queried_city"].isin(selected_cities)]
daily_f = daily[daily["queried_city"].isin(selected_cities)]

CITY_COLORS = {
    "ha-noi": "#EF553B",
    "ho-chi-minh-city": "#636EFA",
    "gia-lai": "#00CC96",
    "quang-ninh": "#FFA15A",
    "thua-thien-hue": "#AB63FA",
    "bac-ninh": "#19D3F3",
    "lao-cai": "#FF6692",
    "da-nang": "#B6E880",
    "cao-bang": "#FECB52",
}

# ── 1. Daily AQI Range (band chart) ──────────────────────────────────────────
st.subheader("Daily AQI — Min / Avg / Max Range")

fig_range = go.Figure()
band_configs = [
    (0, 50, "rgba(0,228,0,0.06)"),
    (50, 100, "rgba(255,255,0,0.06)"),
    (100, 150, "rgba(255,126,0,0.08)"),
    (150, 200, "rgba(255,0,0,0.08)"),
    (200, 300, "rgba(143,63,151,0.08)"),
]
for lo, hi, fill in band_configs:
    fig_range.add_hrect(y0=lo, y1=hi, fillcolor=fill, line_width=0)

for city in selected_cities:
    grp = daily_f[daily_f["queried_city"] == city].sort_values("date")
    color = CITY_COLORS.get(city, "#888")
    label = CITY_LABELS[city]
    # Shaded range (min–max)
    fig_range.add_trace(go.Scatter(
        x=list(grp["date"]) + list(grp["date"][::-1]),
        y=list(grp["max_aqi"]) + list(grp["min_aqi"][::-1]),
        fill="toself",
        fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if "rgb" in color else f"{color}1F",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Daily avg line
    fig_range.add_trace(go.Scatter(
        x=grp["date"],
        y=grp["avg_aqi"],
        mode="lines",
        name=label,
        line=dict(color=color, width=2),
        hovertemplate=f"{label}<br>%{{x|%b %d}}<br>Avg AQI: %{{y:.0f}}<extra></extra>",
    ))

fig_range.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=False, title=""),
    yaxis=dict(showgrid=True, gridcolor="#222", title="AQI"),
    legend=dict(orientation="h", y=1.02),
    height=380, hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_range, use_container_width=True)

# ── 2. Hourly Pattern Heatmap ─────────────────────────────────────────────────
st.subheader("Average AQI by Hour of Day")

hourly = (
    raw_f.groupby(["queried_city", "hour"])["aqi"]
    .mean()
    .reset_index()
)
hourly["city_label"] = hourly["queried_city"].map(CITY_LABELS)

fig_hour = px.line(
    hourly,
    x="hour",
    y="aqi",
    color="city_label",
    color_discrete_map={CITY_LABELS[c]: CITY_COLORS.get(c, "#888") for c in CITY_COLORS},
    labels={"hour": "Hour of Day (local)", "aqi": "Avg AQI", "city_label": "City"},
    markers=True,
)
fig_hour.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(tickmode="linear", dtick=2, showgrid=False, title="Hour (0–23)"),
    yaxis=dict(showgrid=True, gridcolor="#222"),
    legend=dict(title=""),
    height=320, margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_hour, use_container_width=True)

# ── 3. Monthly Average Heatmap ────────────────────────────────────────────────
st.subheader("Monthly Average AQI — Seasonality")

import pandas as pd

monthly_heat = (
    daily_f.groupby(["city_label", "month"])["avg_aqi"]
    .mean()
    .reset_index()
    .pivot(index="city_label", columns="month", values="avg_aqi")
)
monthly_heat.columns = [
    {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}[c]
    for c in monthly_heat.columns
]

fig_heat = px.imshow(
    monthly_heat,
    color_continuous_scale=[
        [0.00, "#00E400"], [0.20, "#FFFF00"], [0.40, "#FF7E00"],
        [0.60, "#FF0000"], [0.80, "#8F3F97"], [1.00, "#7E0023"],
    ],
    text_auto=".0f",
    aspect="auto",
    labels=dict(x="Month", y="City", color="Avg AQI"),
)
fig_heat.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    coloraxis_colorbar=dict(title="AQI"),
    height=280,
    margin=dict(l=0, r=0, t=10, b=0),
)
fig_heat.update_traces(textfont=dict(size=11))
st.plotly_chart(fig_heat, use_container_width=True)

st.caption(
    "Dry season (Nov–Apr): strong temperature inversion traps pollutants — "
    "AQI is significantly higher in the north (Ha Noi, Cao Bang)."
)

# ── 4. Dominant Pollutant Distribution ───────────────────────────────────────
st.subheader("Dominant Pollutant Distribution")
if "dominant_pollutant" in raw_f.columns:
    poll_counts = (
        raw_f.groupby(["city_label", "dominant_pollutant"])
        .size()
        .reset_index(name="count")
    )
    fig_poll = px.bar(
        poll_counts,
        x="city_label",
        y="count",
        color="dominant_pollutant",
        barmode="stack",
        labels={"city_label": "City", "count": "Readings", "dominant_pollutant": "Pollutant"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_poll.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#ccc"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#222"),
        legend=dict(title="Pollutant"),
        height=320, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_poll, use_container_width=True)
