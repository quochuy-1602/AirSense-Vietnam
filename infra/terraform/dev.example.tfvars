aws_region     = "ap-southeast-2"
environment    = "dev"
project_prefix = "airsense"

# Secret — pass via `TF_VAR_waqi_api_token` env var instead of committing.
# waqi_api_token = "..."

alert_email = "you@example.com"

cities = ["ha-noi", "ho-chi-minh-city", "da-nang", "gia-lai", "cao-bang"]

dq_sample_rows              = 10000
anomaly_alert_aqi_threshold = 150
