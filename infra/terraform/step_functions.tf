locals {
  sfn_substitutions = {
    ACCOUNT_ID              = local.account_id
    AWS_REGION              = local.region
    BRONZE_TO_SILVER_API    = aws_glue_job.bronze_to_silver_api.name
    SILVER_TO_GOLD_JOB      = aws_glue_job.silver_to_gold_analytics.name
    SILVER_TO_ML_FEATURES   = aws_glue_job.silver_to_ml_features.name
    DQ_LAMBDA_ARN           = aws_lambda_function.dq_check.arn
    ML_INFERENCE_LAMBDA_ARN = aws_lambda_function.ml_inference.arn
    SNS_TOPIC_ARN           = aws_sns_topic.alerts.arn
    SAGEMAKER_ROLE_ARN      = aws_iam_role.sagemaker.arn
    ML_BUCKET               = aws_s3_bucket.medallion["ml"].bucket
  }
}

resource "aws_sfn_state_machine" "etl" {
  name     = "${local.name_prefix}-etl-pipeline"
  role_arn = aws_iam_role.sfn.arn

  definition = templatefile(
    "${path.module}/../../step_functions/pipeline_orchestation.json",
    local.sfn_substitutions
  )
}

resource "aws_sfn_state_machine" "ml" {
  name     = "${local.name_prefix}-ml-pipeline"
  role_arn = aws_iam_role.sfn.arn

  definition = templatefile(
    "${path.module}/../../step_functions/ml_pipeline.json",
    local.sfn_substitutions
  )
}
