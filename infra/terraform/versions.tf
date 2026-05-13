terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }

  # Uncomment and configure for remote state in production.
  # backend "s3" {
  #   bucket         = "airsense-terraform-state"
  #   key            = "airsense/terraform.tfstate"
  #   region         = "ap-southeast-2"
  #   dynamodb_table = "airsense-terraform-locks"
  #   encrypt        = true
  # }
}
