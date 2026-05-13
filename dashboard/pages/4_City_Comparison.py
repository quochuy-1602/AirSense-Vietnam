"""
Page 4 — City Comparison & Ranking

  • Monthly AQI heatmap (city × month)
  • Monthly ranking table with trend arrows
  • AQI distribution box plots per city
  • Pollution level breakdown (stacked bar)
"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_loader import (
    CITY_LABELS,
    get_daily_summary,
    get_monthly_summary,
)

st.set_page_config(page_title="City Comparison — AirSense", layout="wide")
st.title("🏙️ City Comparison & Ranking")

daily = get_daily_summary()
monthly = get_monthly_summary()

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

# ── 1. Monthly AQI Heatmap ────────────────────────────────────────────────────
st.subheader("Monthly Average AQI — City × Month Heatmap")

pivot = (
    monthly.groupby(["city_label", "month"])["avg_aqi"]
    .mean()
    .reset_index()
    .pivot(index="city_label", columns="month", values="avg_aqi")
)
pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

fig_heat = px.imshow(
    pivot,
    color_continuous_scale=[
        [0.00, "#00E400"], [0.15, "#FFFF00"], [0.35, "#FF7E00"],
        [0.55, "#FF0000"], [0.75, "#8F3F97"], [1.00, "#7E0023"],
    ],
    text_auto=".0f",
    aspect="auto",
    labels=dict(x="Month", y="City", color="Avg AQI"),
    zmin=0, zmax=220,
)
fig_heat.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    coloraxis_colorbar=dict(title="AQI"),
    height=300, margin=dict(l=0, r=0, t=10, b=0),
)
fig_heat.update_traces(textfont=dict(size=11))
st.plotly_chart(fig_heat, use_container_width=True)
st.caption(
    "Clear dry-season (Nov–Apr) pattern in Ha Noi and Cao Bang. "
    "Ho Chi Minh City has higher baseline due to traffic but less seasonal variation."
)

# ── 2. Monthly Ranking Table ──────────────────────────────────────────────────
st.subheader("Monthly City Ranking (Avg AQI, 1 = Most Polluted)")

month_sel = st.selectbox(
    "Select month",
    options=list(range(1, 13)),
    format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
    index=0,
)

month_data = (
    monthly[monthly["month"] == month_sel]
    .sort_values("aqi_rank")[
        ["aqi_rank", "city_label", "avg_aqi", "max_aqi", "pollution_level"]
    ]
    .copy()
)

if "avg_pm25" in monthly.columns:
    month_data = monthly[monthly["month"] == month_sel].sort_values("aqi_rank")[
        ["aqi_rank", "city_label", "avg_aqi", "avg_pm25", "max_aqi", "pollution_level"]
    ].copy()

month_data.columns = (
    ["Rank", "City", "Avg AQI", "Avg PM2.5", "Max AQI", "Level"]
    if len(month_data.columns) == 6
    else ["Rank", "City", "Avg AQI", "Max AQI", "Level"]
)
month_data = month_data.round(1)

def _level_color(val):
    color_map = {
        "Good": "#00E400", "Moderate": "#BFBF00",
        "Unhealthy for Sensitive": "#FF7E00",
        "Unhealthy": "#FF0000", "Very Unhealthy": "#8F3F97",
        "Hazardous": "#7E0023",
    }
    c = color_map.get(str(val), "")
    return f"color:{c};font-weight:bold" if c else ""

st.dataframe(
    month_data.style.applymap(_level_color, subset=["Level"]),
    use_container_width=True, hide_index=True,
)

# ── 3. AQI Distribution Box Plots ────────────────────────────────────────────
st.subheader("AQI Distribution per City")
st.caption("Box plots show median, IQR, and outliers for daily average AQI.")

fig_box = go.Figure()
for city in sorted(daily["queried_city"].unique()):
    values = daily[daily["queried_city"] == city]["avg_aqi"].dropna()
    fig_box.add_trace(go.Box(
        y=values,
        name=CITY_LABELS.get(city, city),
        marker_color=CITY_COLORS.get(city, "#888"),
        boxpoints="outliers",
        jitter=0.3,
        hovertemplate="AQI: %{y:.0f}<extra></extra>",
    ))

fig_box.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#222", title="Daily Avg AQI"),
    showlegend=False,
    height=360, margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_box, use_container_width=True)

# ── 4. Pollution Level Breakdown ──────────────────────────────────────────────
st.subheader("Days by Pollution Level per City")

level_order = [
    "Good", "Moderate", "Unhealthy for Sensitive",
    "Unhealthy", "Very Unhealthy", "Hazardous",
]
level_colors = {
    "Good": "#00E400", "Moderate": "#FFFF00",
    "Unhealthy for Sensitive": "#FF7E00",
    "Unhealthy": "#FF0000", "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
}

level_counts = (
    daily.groupby(["city_label", "pollution_level"])
    .size()
    .reset_index(name="days")
)

fig_stacked = px.bar(
    level_counts,
    x="city_label",
    y="days",
    color="pollution_level",
    barmode="stack",
    category_orders={"pollution_level": level_order},
    color_discrete_map=level_colors,
    labels={"city_label": "City", "days": "Days", "pollution_level": "Level"},
)
fig_stacked.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=False, title=""),
    yaxis=dict(showgrid=True, gridcolor="#222", title="Days in 2021"),
    legend=dict(title="Pollution Level", orientation="v"),
    height=360, margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_stacked, use_container_width=True)

# ── 5. Year-round Summary Table ───────────────────────────────────────────────
st.subheader("Annual Summary — 2021")

annual = (
    daily.groupby(["queried_city", "city_label"])
    .agg(
        avg_aqi=("avg_aqi", "mean"),
        max_aqi=("max_aqi", "max"),
        days_good=("pollution_level", lambda x: (x == "Good").sum()),
        days_unhealthy=("pollution_level", lambda x: x.isin(["Unhealthy", "Very Unhealthy", "Hazardous"]).sum()),
    )
    .reset_index()
    .sort_values("avg_aqi", ascending=False)
)
annual["Overall Rank"] = range(1, len(annual) + 1)
annual = annual.rename(columns={
    "city_label": "City", "avg_aqi": "Avg AQI", "max_aqi": "Max AQI (day)",
    "days_good": "Good Days", "days_unhealthy": "Unhealthy+ Days",
}).drop(columns=["queried_city"]).round(1)

st.dataframe(annual.set_index("Overall Rank"), use_container_width=True)
