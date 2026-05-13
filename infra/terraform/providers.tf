provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "AirSense-Vietnam"
      ManagedBy   = "Terraform"
      Environment = var.environment
    }
  }
}
