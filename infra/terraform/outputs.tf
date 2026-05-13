output "bucket_names" {
  description = "S3 bucket names for each medallion layer."
  value       = { for k, b in aws_s3_bucket.medallion : k => b.bucket }
}

output "glue_databases" {
  description = "Glue catalog database names."
  value       = { for k, db in aws_glue_catalog_database.medallion : k => db.name }
}

output "sns_topic_arn" {
  description = "SNS topic for pipeline alerts."
  value       = aws_sns_topic.alerts.arn
}

output "state_machine_arns" {
  description = "Step Functions state machine ARNs."
  value = {
    etl = aws_sfn_state_machine.etl.arn
    ml  = aws_sfn_state_machine.ml.arn
  }
}

output "lambda_function_names" {
  value = {
    ingestion    = aws_lambda_function.ingestion.function_name
    dq_check     = aws_lambda_function.dq_check.function_name
    ml_inference = aws_lambda_function.ml_inference.function_name
  }
}
