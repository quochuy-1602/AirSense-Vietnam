"""
AirSense Vietnam — Air Quality Intelligence Dashboard

Main overview page: city KPI cards + rolling AQI trend.
"""
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_loader import (
    AQI_LEVELS,
    CITY_COORDS,
    CITY_LABELS,
    get_aqi_level,
    get_daily_summary,
    load_raw,
)

st.set_page_config(
    page_title="AirSense Vietnam",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌿 AirSense Vietnam")
st.caption(
    "End-to-end air quality intelligence platform — real-time ingestion, "
    "XGBoost 24h forecasting, anomaly detection, and SHAP explainability."
)
st.divider()

# ── Data ──────────────────────────────────────────────────────────────────────
daily = get_daily_summary()
latest = (
    daily.sort_values("date")
    .groupby("queried_city")
    .last()
    .reset_index()
)

# ── AQI Level Legend ──────────────────────────────────────────────────────────
st.subheader("AQI Scale Reference")
cols_leg = st.columns(len(AQI_LEVELS))
for col, (lo, hi, label, color) in zip(cols_leg, AQI_LEVELS):
    col.markdown(
        f"""<div style='background:{color};padding:8px 4px;border-radius:6px;
        text-align:center;color:#000;font-size:11px;font-weight:600'>
        {label}<br><span style='font-size:10px'>{lo}–{hi if hi < 9999 else "500+"}</span>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# ── City KPI Cards ────────────────────────────────────────────────────────────
st.subheader("Latest Readings by City")
kpi_cols = st.columns(len(latest))
for col, (_, row) in zip(kpi_cols, latest.iterrows()):
    label, color = get_aqi_level(row["avg_aqi"])
    trend = "↑" if row.get("avg_aqi", 0) > 100 else "↓"
    col.markdown(
        f"""<div style='border:1px solid #333;border-radius:10px;padding:14px 10px;
        background:#1a1a2e;text-align:center'>
        <div style='font-size:13px;color:#aaa'>{CITY_LABELS.get(row['queried_city'], row['queried_city'])}</div>
        <div style='font-size:36px;font-weight:700;color:{color}'>{row['avg_aqi']:.0f}</div>
        <div style='font-size:12px;color:{color}'>{label}</div>
        <div style='font-size:11px;color:#888;margin-top:4px'>PM2.5: {row.get("avg_pm25", 0):.1f} μg/m³</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("&nbsp;")

# ── Vietnam Map ───────────────────────────────────────────────────────────────
st.subheader("Pollution Map — Vietnam")
map_data = []
for _, row in latest.iterrows():
    city = row["queried_city"]
    if city in CITY_COORDS:
        lat, lon = CITY_COORDS[city]
        label, color = get_aqi_level(row["avg_aqi"])
        map_data.append(
            dict(
                city=CITY_LABELS.get(city, city),
                lat=lat,
                lon=lon,
                aqi=row["avg_aqi"],
                level=label,
                color=color,
            )
        )

fig_map = go.Figure()
for d in map_data:
    fig_map.add_trace(
        go.Scattergeo(
            lon=[d["lon"]],
            lat=[d["lat"]],
            mode="markers+text",
            marker=dict(size=d["aqi"] / 5 + 12, color=d["color"], opacity=0.85,
                        line=dict(width=1, color="#fff")),
            text=d["city"],
            textposition="top center",
            hovertemplate=(
                f"<b>{d['city']}</b><br>AQI: {d['aqi']:.0f}<br>"
                f"Level: {d['level']}<extra></extra>"
            ),
            showlegend=False,
        )
    )
fig_map.update_layout(
    geo=dict(
        scope="asia",
        center=dict(lat=16, lon=106),
        projection_scale=7,
        showland=True,
        landcolor="#1e1e2e",
        showocean=True,
        oceancolor="#0d1117",
        showcoastlines=True,
        coastlinecolor="#444",
        bgcolor="#0d1117",
    ),
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    height=420,
    margin=dict(l=0, r=0, t=0, b=0),
)
st.plotly_chart(fig_map, use_container_width=True)

# ── Rolling 30-day AQI Trend — all cities ────────────────────────────────────
st.subheader("2021 AQI Trends — All Cities")

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

fig_trend = go.Figure()
for city, grp in daily.groupby("queried_city"):
    grp = grp.sort_values("date")
    rolling_avg = grp["avg_aqi"].rolling(7, min_periods=1).mean()
    fig_trend.add_trace(
        go.Scatter(
            x=grp["date"],
            y=rolling_avg,
            mode="lines",
            name=CITY_LABELS.get(city, city),
            line=dict(color=CITY_COLORS.get(city, "#888"), width=2),
            hovertemplate="%{x|%b %d}<br>AQI: %{y:.0f}<extra></extra>",
        )
    )

# AQI level background bands
band_configs = [
    (0, 50, "rgba(0,228,0,0.06)"),
    (50, 100, "rgba(255,255,0,0.06)"),
    (100, 150, "rgba(255,126,0,0.08)"),
    (150, 200, "rgba(255,0,0,0.08)"),
    (200, 300, "rgba(143,63,151,0.08)"),
]
for lo, hi, fill in band_configs:
    fig_trend.add_hrect(y0=lo, y1=hi, fillcolor=fill, line_width=0)

fig_trend.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=False, title=""),
    yaxis=dict(showgrid=True, gridcolor="#222", title="AQI (7-day rolling avg)"),
    legend=dict(orientation="h", y=1.02, x=0),
    height=360,
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
)
st.plotly_chart(fig_trend, use_container_width=True)

# ── Pipeline Architecture Note ────────────────────────────────────────────────
st.divider()
with st.expander("Pipeline Architecture", expanded=False):
    st.markdown("""
```
WAQI API  ──► Bronze (S3 raw)
               │
               ▼  [Glue bronze→silver]
            Silver (fact_aqi)
               │
       ┌───────┴──────────┐
       ▼                  ▼
[silver→ml_features]  [silver→gold]
   ML Features            Gold Analytics
       │                  │
       ▼                  ▼
SageMaker Training    QuickSight / This Dashboard
 XGBoost + IF
       │
  Model Registry
       │
SageMaker Batch Transform
(daily inference)
       │
  Gold: predictions
        + anomalies
```
""")
    st.caption("Orchestrated by AWS Step Functions · IaC via Terraform · Monitored via CloudWatch")
