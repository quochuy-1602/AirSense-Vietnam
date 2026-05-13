variable "aws_region" {
  description = "AWS region to deploy resources in."
  type        = string
  default     = "ap-southeast-2"
}

variable "environment" {
  description = "Deployment environment (dev / staging / prod)."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "project_prefix" {
  description = "Prefix for all resource names (keeps things namespaced)."
  type        = string
  default     = "airsense"
}

variable "waqi_api_token" {
  description = "WAQI API token. Pass via TF_VAR_waqi_api_token or a tfvars file — never commit."
  type        = string
  sensitive   = true
  default     = ""
}

variable "alert_email" {
  description = "Email address to subscribe to pipeline alerts SNS topic."
  type        = string
  default     = ""
}

variable "cities" {
  description = "List of Vietnamese cities to ingest."
  type        = list(string)
  default     = ["ha-noi", "ho-chi-minh-city", "da-nang", "gia-lai", "cao-bang"]
}

variable "ingestion_schedules" {
  description = "EventBridge cron schedules (UTC) for the ingestion Lambda."
  type        = map(string)
  default = {
    morning   = "cron(0 1 * * ? *)" # 08:00 ICT
    afternoon = "cron(0 7 * * ? *)" # 14:00 ICT
    evening   = "cron(0 13 * * ? *)" # 20:00 ICT
  }
}

variable "dq_sample_rows" {
  description = "Number of rows to sample in DQ Lambda (Athena)."
  type        = number
  default     = 10000
}

variable "anomaly_alert_aqi_threshold" {
  description = "AQI threshold above which anomaly detection publishes an SNS alert."
  type        = number
  default     = 150
}
