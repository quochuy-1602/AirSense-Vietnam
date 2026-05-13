data "archive_file" "ingestion_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../lambdas/air-quality-api-ingestion"
  output_path = "${path.module}/.build/ingestion.zip"
}

data "archive_file" "dq_check_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../lambdas/quality_data"
  output_path = "${path.module}/.build/dq_check.zip"
}

data "archive_file" "ml_inference_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../lambdas/ml_inference"
  output_path = "${path.module}/.build/ml_inference.zip"
}

resource "aws_lambda_function" "ingestion" {
  function_name    = "${local.name_prefix}-waqi-ingestion"
  role             = aws_iam_role.lambda.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.11"
  timeout          = 60
  memory_size      = 256
  filename         = data.archive_file.ingestion_zip.output_path
  source_code_hash = data.archive_file.ingestion_zip.output_base64sha256

  environment {
    variables = {
      WAQI_API_TOKEN      = var.waqi_api_token
      S3_BUCKET_BRONZE    = aws_s3_bucket.medallion["bronze"].bucket
      WAQI_CITIES         = join(",", var.cities)
      SNS_ALERT_TOPIC_ARN = aws_sns_topic.alerts.arn
    }
  }
}

resource "aws_lambda_function" "dq_check" {
  function_name    = "${local.name_prefix}-dq-check"
  role             = aws_iam_role.lambda.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.11"
  timeout          = 300
  memory_size      = 512
  filename         = data.archive_file.dq_check_zip.output_path
  source_code_hash = data.archive_file.dq_check_zip.output_base64sha256

  environment {
    variables = {
      ATHENA_DATABASE         = local.glue_databases.silver
      ATHENA_OUTPUT_LOCATION  = "s3://${aws_s3_bucket.medallion["athena"].bucket}/results/"
      SNS_ALERT_TOPIC_ARN     = aws_sns_topic.alerts.arn
      DQ_SAMPLE_ROWS          = tostring(var.dq_sample_rows)
    }
  }
}

resource "aws_lambda_function" "ml_inference" {
  function_name    = "${local.name_prefix}-ml-inference"
  role             = aws_iam_role.lambda.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.11"
  timeout          = 900
  memory_size      = 512
  filename         = data.archive_file.ml_inference_zip.output_path
  source_code_hash = data.archive_file.ml_inference_zip.output_base64sha256

  environment {
    variables = {
      ML_FEATURES_BUCKET          = aws_s3_bucket.medallion["ml"].bucket
      ML_FEATURES_PREFIX          = "features/aqi_features/"
      BATCH_OUTPUT_BUCKET         = aws_s3_bucket.medallion["ml"].bucket
      GOLD_BUCKET                 = aws_s3_bucket.medallion["gold"].bucket
      GOLD_DATABASE               = local.glue_databases.gold
      FORECAST_MODEL_PKG_GROUP    = "aqi-forecast-models"
      ANOMALY_MODEL_PKG_GROUP     = "aqi-anomaly-models"
      SAGEMAKER_ROLE_ARN          = aws_iam_role.sagemaker.arn
      SNS_ALERT_TOPIC_ARN         = aws_sns_topic.alerts.arn
      ANOMALY_ALERT_AQI_THRESHOLD = tostring(var.anomaly_alert_aqi_threshold)
    }
  }
}

# Allow EventBridge to invoke the ingestion Lambda
resource "aws_lambda_permission" "events_invoke_ingestion" {
  for_each      = var.ingestion_schedules
  statement_id  = "AllowEventBridge${each.key}"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingestion.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.ingestion[each.key].arn
}

resource "aws_cloudwatch_event_rule" "ingestion" {
  for_each            = var.ingestion_schedules
  name                = "${local.name_prefix}-ingest-${each.key}"
  schedule_expression = each.value
}

resource "aws_cloudwatch_event_target" "ingestion" {
  for_each = var.ingestion_schedules
  rule     = aws_cloudwatch_event_rule.ingestion[each.key].name
  arn      = aws_lambda_function.ingestion.arn
}
