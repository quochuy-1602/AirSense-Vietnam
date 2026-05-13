"""
Page 2 — 24h AQI Forecast + SHAP Explainability

• Predicted vs Actual time series (test set: Oct–Dec 2021)
• Model performance metrics (MAE, RMSE, MAPE)
• SHAP feature importance bar chart
• SHAP scatter: feature value vs SHAP contribution (top 6 features)
• Single-prediction explanation (waterfall-style)
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.data_loader import CITY_LABELS
from dashboard.utils.ml_utils import SHAP_FEATURE_LABELS, get_forecast_bundle

st.set_page_config(page_title="Forecast & SHAP — AirSense", layout="wide")
st.title("🔮 24h AQI Forecast + SHAP Explainability")

st.info(
    "**Model:** XGBoost trained on Jan–Sep 2021 · evaluated on Oct–Dec 2021.  \n"
    "**SHAP** (SHapley Additive exPlanations) explains *why* each prediction is high or low — "
    "required for regulatory transparency in real-world deployments.",
    icon="ℹ️",
)

with st.spinner("Training model and computing SHAP values (first load only)…"):
    bundle = get_forecast_bundle()

metrics = bundle["metrics"]
test_df = bundle["test_df"]
shap_importance = bundle["shap_importance"]
shap_values = bundle["shap_values"]
X_shap = bundle["X_shap"]
feature_names = bundle["feature_names"]

# ── 1. Model Metrics ──────────────────────────────────────────────────────────
st.subheader("Model Performance — Test Set (Oct–Dec 2021)")
m_cols = st.columns(3)
for col, (k, v) in zip(m_cols, metrics.items()):
    col.metric(k, f"{v:.2f}")

# ── 2. Predicted vs Actual ────────────────────────────────────────────────────
st.subheader("Predicted vs Actual AQI — All Cities")

city_filter = st.selectbox(
    "Filter by city",
    ["All cities"] + list(CITY_LABELS.keys()),
    format_func=lambda c: "All cities" if c == "All cities" else CITY_LABELS[c],
)

plot_df = test_df if city_filter == "All cities" else test_df[test_df["queried_city"] == city_filter]
plot_df = plot_df.sort_values("measured_at")

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(
    x=plot_df["measured_at"], y=plot_df["actual"],
    mode="lines", name="Actual AQI",
    line=dict(color="#636EFA", width=1.5),
    hovertemplate="%{x|%b %d %H:%M}<br>Actual: %{y:.0f}<extra></extra>",
))
fig_pred.add_trace(go.Scatter(
    x=plot_df["measured_at"], y=plot_df["predicted"],
    mode="lines", name="Predicted (24h ahead)",
    line=dict(color="#EF553B", width=1.5, dash="dash"),
    hovertemplate="%{x|%b %d %H:%M}<br>Predicted: %{y:.0f}<extra></extra>",
))
fig_pred.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=False, title=""),
    yaxis=dict(showgrid=True, gridcolor="#222", title="AQI"),
    legend=dict(orientation="h", y=1.02),
    height=340, hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_pred, use_container_width=True)

# ── 3. SHAP Feature Importance ────────────────────────────────────────────────
st.subheader("SHAP Feature Importance (Mean |SHAP value|)")
st.caption(
    "Higher SHAP importance = feature has larger average impact on predictions. "
    "Rolling averages dominate, confirming the strong temporal autocorrelation of AQI."
)

fig_imp = go.Figure(go.Bar(
    x=shap_importance["mean_abs_shap"].head(15)[::-1],
    y=shap_importance["label"].head(15)[::-1],
    orientation="h",
    marker=dict(
        color=shap_importance["mean_abs_shap"].head(15)[::-1],
        colorscale="Reds",
        showscale=False,
    ),
    hovertemplate="%{y}<br>Mean |SHAP|: %{x:.3f}<extra></extra>",
))
fig_imp.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=True, gridcolor="#222", title="Mean |SHAP value|"),
    yaxis=dict(showgrid=False),
    height=420, margin=dict(l=0, r=20, t=10, b=0),
)
st.plotly_chart(fig_imp, use_container_width=True)

# ── 4. SHAP Scatter — Feature Value vs Contribution ──────────────────────────
st.subheader("SHAP Detail — Feature Value vs Contribution (top 6)")
st.caption(
    "Each dot is one test sample. Color = feature value (blue=low, red=high). "
    "Positive SHAP → pushes prediction higher than base; negative → pushes it lower."
)

top6 = shap_importance["feature"].head(6).tolist()
n_cols = 3
rows = [top6[i:i+n_cols] for i in range(0, len(top6), n_cols)]

for row_feats in rows:
    row_cols = st.columns(n_cols)
    for col, feat in zip(row_cols, row_feats):
        feat_idx = feature_names.index(feat)
        feat_vals = X_shap[:, feat_idx]
        shap_contrib = shap_values[:, feat_idx]

        fig_s = go.Figure(go.Scatter(
            x=feat_vals,
            y=shap_contrib,
            mode="markers",
            marker=dict(
                color=feat_vals,
                colorscale="RdBu_r",
                size=4,
                opacity=0.6,
                showscale=False,
            ),
            hovertemplate=f"Feature value: %{{x:.2f}}<br>SHAP: %{{y:.3f}}<extra></extra>",
        ))
        fig_s.add_hline(y=0, line_width=1, line_color="#555")
        fig_s.update_layout(
            title=dict(text=SHAP_FEATURE_LABELS.get(feat, feat), font=dict(size=12)),
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#ccc", size=10),
            xaxis=dict(showgrid=False, title="Feature value"),
            yaxis=dict(showgrid=True, gridcolor="#222", title="SHAP"),
            height=220, margin=dict(l=30, r=10, t=30, b=30),
        )
        col.plotly_chart(fig_s, use_container_width=True)

# ── 5. Single Prediction Explanation ─────────────────────────────────────────
st.subheader("Explain a Single Prediction (Waterfall)")
st.caption("Select a sample index from the test set to see how each feature contributed to that prediction.")

sample_idx = st.slider("Sample index (test set)", 0, len(X_shap) - 1, 0)

base_value = float(np.mean(test_df["actual"].values[:len(X_shap)]))
sample_shap = shap_values[sample_idx]
sample_feat = X_shap[sample_idx]

# Build waterfall data — top 10 by absolute SHAP
sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:10]
labels_wf = [SHAP_FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in sorted_idx]
values_wf = [float(sample_shap[i]) for i in sorted_idx]
feat_vals_wf = [float(sample_feat[i]) for i in sorted_idx]

predicted_val = base_value + sum(values_wf)

fig_wf = go.Figure(go.Waterfall(
    name="SHAP contributions",
    orientation="h",
    measure=["relative"] * len(values_wf) + ["total"],
    y=labels_wf + ["Prediction"],
    x=values_wf + [0],
    base=base_value,
    connector=dict(line=dict(color="#555", width=0.5)),
    decreasing=dict(marker=dict(color="#636EFA")),
    increasing=dict(marker=dict(color="#EF553B")),
    totals=dict(marker=dict(color="#FFA15A")),
    hovertemplate="%{y}<br>SHAP: %{x:.2f}<extra></extra>",
))
fig_wf.add_vline(x=base_value, line_width=1, line_dash="dot", line_color="#888",
                  annotation_text=f"Base: {base_value:.0f}", annotation_position="top right")
fig_wf.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#ccc"),
    xaxis=dict(showgrid=True, gridcolor="#222", title="AQI contribution"),
    yaxis=dict(showgrid=False),
    height=400, margin=dict(l=10, r=10, t=10, b=0),
)
st.plotly_chart(fig_wf, use_container_width=True)

actual_val = test_df["actual"].values[sample_idx] if sample_idx < len(test_df) else "—"
pred_val = test_df["predicted"].values[sample_idx] if sample_idx < len(test_df) else "—"
st.caption(
    f"Sample {sample_idx} — Actual AQI: **{actual_val:.0f}** · "
    f"Model prediction: **{pred_val:.0f}** · "
    f"SHAP sum from base ({base_value:.0f}): **{sum(values_wf):+.0f}**"
    if isinstance(actual_val, (int, float, np.floating)) else ""
)
