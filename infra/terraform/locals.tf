data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name

  name_prefix = "${var.project_prefix}-${var.environment}"

  bucket_names = {
    bronze = "${local.name_prefix}-bronze-${local.account_id}"
    silver = "${local.name_prefix}-silver-${local.account_id}"
    gold   = "${local.name_prefix}-gold-${local.account_id}"
    ml     = "${local.name_prefix}-ml-${local.account_id}"
    athena = "${local.name_prefix}-athena-results-${local.account_id}"
    code   = "${local.name_prefix}-code-${local.account_id}"
  }

  glue_databases = {
    bronze = "${local.name_prefix}-bronze"
    silver = "${local.name_prefix}-silver"
    gold   = "${local.name_prefix}-gold"
    ml     = "${local.name_prefix}-ml"
  }
}
