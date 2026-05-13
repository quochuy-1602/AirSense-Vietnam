# AirSense Vietnam — Dashboard

Interactive Streamlit dashboard for the Vietnam Air Quality Data Pipeline.

## Pages

| Page | Description |
|---|---|
| **Overview** | City KPI cards, Vietnam pollution map, 2021 AQI trend |
| **AQI Trends** | Daily range bands, hourly heatmap, seasonal pattern, pollutant breakdown |
| **Forecast & SHAP** | XGBoost next-day prediction vs actual, SHAP feature importance, waterfall explanation |
| **Anomaly Detection** | Isolation Forest anomaly markers, score distribution, alert table |
| **City Comparison** | Monthly heatmap, ranking table, distribution box plots, pollution level breakdown |

## Run locally

```bash
cd <project-root>
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Architecture

- **Data**: reads `raw_data_csv/historical_air_quality_2021_en.csv` by default
- **ML**: trains XGBoost + Isolation Forest on first load; cached in-session via `@st.cache_resource`
- **SHAP**: `TreeExplainer` on 400-sample test set; global importance + per-prediction waterfall
- **Production**: swap `load_raw()` in `utils/data_loader.py` for `awswrangler.s3.read_parquet()` pointing at the Gold S3 layer
