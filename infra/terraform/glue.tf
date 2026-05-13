resource "aws_glue_catalog_database" "medallion" {
  for_each = local.glue_databases
  name     = each.value
}

resource "aws_glue_job" "bronze_to_silver_api" {
  name         = "${local.name_prefix}-bronze-to-silver-api"
  role_arn     = aws_iam_role.glue.arn
  glue_version = "4.0"

  command {
    script_location = "s3://${aws_s3_bucket.medallion["code"].bucket}/glue_jobs/bronze_to_silver_statistics_api.py"
    python_version  = "3"
  }

  default_arguments = {
    "--job-bookmark-option"     = "job-bookmark-enable"
    "--enable-metrics"          = "true"
    "--enable-continuous-cloudwatch-log" = "true"
    "--bronze_database"         = local.glue_databases.bronze
    "--bronze_table"            = "api_raw"
    "--silver_bucket"           = aws_s3_bucket.medallion["silver"].bucket
    "--silver_database"         = local.glue_databases.silver
    "--stale_hours"             = "72"
  }

  max_retries       = 1
  worker_type       = "G.1X"
  number_of_workers = 2
  timeout           = 30
}

resource "aws_glue_job" "silver_to_gold_analytics" {
  name         = "${local.name_prefix}-silver-to-gold-analytics"
  role_arn     = aws_iam_role.glue.arn
  glue_version = "4.0"

  command {
    script_location = "s3://${aws_s3_bucket.medallion["code"].bucket}/glue_jobs/silver_to_gold_analytics.py"
    python_version  = "3"
  }

  default_arguments = {
    "--enable-metrics"  = "true"
    "--silver_database" = local.glue_databases.silver
    "--gold_bucket"     = aws_s3_bucket.medallion["gold"].bucket
    "--gold_database"   = local.glue_databases.gold
  }

  worker_type       = "G.1X"
  number_of_workers = 2
  timeout           = 20
}

resource "aws_glue_job" "silver_to_ml_features" {
  name         = "${local.name_prefix}-silver-to-ml-features"
  role_arn     = aws_iam_role.glue.arn
  glue_version = "4.0"

  command {
    script_location = "s3://${aws_s3_bucket.medallion["code"].bucket}/glue_jobs/silver_to_ml_features.py"
    python_version  = "3"
  }

  default_arguments = {
    "--enable-metrics"       = "true"
    "--silver_database"      = local.glue_databases.silver
    "--silver_table"         = "fact_aqi"
    "--ml_features_bucket"   = aws_s3_bucket.medallion["ml"].bucket
    "--ml_features_database" = local.glue_databases.ml
    "--forecast_horizon_h"   = "24"
  }

  worker_type       = "G.1X"
  number_of_workers = 2
  timeout           = 30
}
