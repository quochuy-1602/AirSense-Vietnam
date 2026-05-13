resource "aws_s3_bucket" "medallion" {
  for_each      = local.bucket_names
  bucket        = each.value
  force_destroy = var.environment != "prod"
}

resource "aws_s3_bucket_versioning" "medallion" {
  for_each = aws_s3_bucket.medallion

  bucket = each.value.id

  versioning_configuration {
    status = var.environment == "prod" ? "Enabled" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "medallion" {
  for_each = aws_s3_bucket.medallion

  bucket = each.value.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "medallion" {
  for_each = aws_s3_bucket.medallion

  bucket = each.value.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "athena_results" {
  bucket = aws_s3_bucket.medallion["athena"].id

  rule {
    id     = "expire-athena-results"
    status = "Enabled"

    filter {}

    expiration {
      days = 7
    }
  }
}
