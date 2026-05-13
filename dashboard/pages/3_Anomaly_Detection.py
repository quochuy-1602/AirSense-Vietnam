"""
Page 3 — Anomaly Detection

Isolation Forest (contamination=2%) applied to the full dataset:
  • Time series with anomaly events highlighted
  • Anomaly score distribution
  • Alert table of the most severe events
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_loader import CITY_LABELS
from dashboard.utils.ml_utils import get_anomaly_bundle

st.set_page_config(page_title="Anomaly Detection — AirSense", layout="wide")
st.title("🚨 Anomaly Detection")

st.info(
    "**Model:** Isolation Forest (n_estimators=200, contamination=2%) trained on the full 2021 dataset.  \n"
    "Anomalies are detected using multivariate patterns across AQI, PM2.5, PM10, temperature, "
    "humidity, pressure, and wind — capturing events that deviate from normal joint distributions.",
    icon="ℹ️",
)

with st.spinner("Running anomaly detection (first load only)…"):
    df = get_anomaly_bundle()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
cities = sorted(df["queried_city"].unique())
selected_city = st.sidebar.selectbox(
    "City", cities, format_func=lambda c: CITY_LABELS.get(c, c)
)

city_df = df[df["queried_city"] == selected_city].sort_values("measured_at")

# ── KPI Summary ───────────────────────────────────────────────────────────────
total = len(city_df)
n_anomaly = int(city_df["is_anomaly"].sum())
rate = n_anomaly / max(total, 1) * 100
high_aqi_anomalies = int((city_df["is_anomaly"] & (city_df["aqi"] > 150)).sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Readings", f"{total:,}")
k2.metric("Anomalies Detected", f"{n_anomaly:,}")
k3.metric("Anomaly Rate", f"{rate:.1f}%")
k4.metric("High-AQI Anomalies (>150)", f"{high_aqi_anomalies:,}")

# ── 1. AQI Time Series with Anomaly Markers ───────────────────────────────────
st.subheader(f"AQI Time Series — {CITY_LABELS.get(selected_city, selected_city)}")

anomaly = city_df[city_df["is_anomaly"]]

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=city_df["measured_at"], y=city_df["aqi"],
    mode="lines", name="AQI",
    line=dict(color="#636EFA", width=1),
    hovertemplate="%{x|%b %d %H:%M}<br>AQI: %{y:.0f}<extra></extra>",
))
fig_ts.add_trace(go.Scatter(
    x=anomaly["measured_at"], y=anomaly["aqi"],
    mode="markers", name="Anomaly",
    marker=dict(color="#EF553B", size=7, symbol="x",
                line=dict(width=1.5, color="#fff")),
    customdata=anomaly["anomaly_score"].values.reshape(-1, 1),
    hovertemplate="%{x|%b %d %H:%M}<br>AQI: %{y:.0f}<br>Score: %{customdata[0]:.3f}<extra></extra>",
))

for lo, hi, fill in [
    (100, 150, "rgba(255,126,0,0.07)"),
    (150, 200, "rgba(255,0,0,0.07)"),
    (200, 9999, "rgba(143,63,151,0.08)"),
]:
    fig_ts.add_hrect(y0=lo, y1=hi, fillcolor=fill, line_width=0)

fig_ts.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=False, title=""),
    yaxis=dict(showgrid=True, gridcolor="#222", title="AQI"),
    legend=dict(orientation="h", y=1.02),
    height=340, hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_ts, use_container_width=True)

# ── 2. Anomaly Score Distribution ────────────────────────────────────────────
st.subheader("Anomaly Score Distribution")
st.caption(
    "Higher anomaly score = more isolated in feature space. "
    "Red line marks the threshold separating normal from anomalous."
)

threshold = anomaly["anomaly_score"].min() if n_anomaly > 0 else None
scores_normal = city_df[~city_df["is_anomaly"]]["anomaly_score"].dropna().values
scores_anom = city_df[city_df["is_anomaly"]]["anomaly_score"].dropna().values

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(
    x=scores_normal, name="Normal", opacity=0.7,
    marker_color="#636EFA", nbinsx=60,
))
if len(scores_anom) > 0:
    fig_dist.add_trace(go.Histogram(
        x=scores_anom, name="Anomaly", opacity=0.85,
        marker_color="#EF553B", nbinsx=40,
    ))
if threshold:
    fig_dist.add_vline(x=threshold, line_width=2, line_dash="dash",
                        line_color="#FF7E00",
                        annotation_text="Threshold",
                        annotation_position="top right")

fig_dist.update_layout(
    barmode="overlay",
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=False, title="Anomaly Score"),
    yaxis=dict(showgrid=True, gridcolor="#222", title="Count"),
    legend=dict(orientation="h", y=1.02),
    height=300, margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_dist, use_container_width=True)

# ── 3. Alert Table ────────────────────────────────────────────────────────────
st.subheader("Top Anomaly Events — Most Severe")

display_cols = [c for c in ["measured_at", "aqi", "pm25", "pm10",
                             "temperature", "humidity", "anomaly_score"]
                if c in city_df.columns]
top_alerts = (
    city_df[city_df["is_anomaly"]]
    .sort_values("anomaly_score", ascending=False)
    .head(20)[display_cols]
    .copy()
)

if not top_alerts.empty:
    top_alerts["measured_at"] = top_alerts["measured_at"].dt.strftime("%Y-%m-%d %H:%M")
    col_rename = {
        "measured_at": "Timestamp", "aqi": "AQI", "pm25": "PM2.5",
        "pm10": "PM10", "temperature": "Temp (°C)",
        "humidity": "Humidity (%)", "anomaly_score": "Anomaly Score",
    }
    top_alerts = top_alerts.rename(columns=col_rename).round(2)

    def _color_aqi(val):
        try:
            v = float(val)
            if v > 200: return "color:#8F3F97;font-weight:bold"
            if v > 150: return "color:#FF0000;font-weight:bold"
            if v > 100: return "color:#FF7E00"
        except Exception:
            pass
        return ""

    st.dataframe(
        top_alerts.style
            .applymap(_color_aqi, subset=["AQI"])
            .background_gradient(subset=["Anomaly Score"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write("No anomalies detected for this city.")

# ── 4. Monthly Anomaly Count ──────────────────────────────────────────────────
st.subheader("Monthly Anomaly Count")

month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

monthly_anom = (
    city_df[city_df["is_anomaly"]]
    .assign(month=city_df["measured_at"].dt.month)
    .groupby("month").size()
    .reindex(range(1, 13), fill_value=0)
    .reset_index()
)
monthly_anom.columns = ["Month", "Anomaly Count"]
monthly_anom["Month"] = monthly_anom["Month"].map(month_names)

fig_bar = px.bar(
    monthly_anom, x="Month", y="Anomaly Count",
    color="Anomaly Count", color_continuous_scale="Reds",
)
fig_bar.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#222"),
    coloraxis_showscale=False,
    height=260, margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_bar, use_container_width=True)
st.caption(
    "Winter months (Nov–Feb) show elevated anomaly counts driven by "
    "thermal inversion layers trapping pollutants in northern Vietnam."
)
