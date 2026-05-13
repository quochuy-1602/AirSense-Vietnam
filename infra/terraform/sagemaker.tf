resource "aws_sagemaker_model_package_group" "forecast" {
  model_package_group_name        = "aqi-forecast-models"
  model_package_group_description = "AQI forecasting models (XGBoost)"
}

resource "aws_sagemaker_model_package_group" "anomaly" {
  model_package_group_name        = "aqi-anomaly-models"
  model_package_group_description = "AQI anomaly detection models (Isolation Forest)"
}
