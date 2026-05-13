# Terraform — AirSense Vietnam Infrastructure

Provisions the full AWS infrastructure for the AirSense pipeline:

- **S3** — 6 buckets (bronze, silver, gold, ml, athena-results, code)
- **Glue** — 4 catalog databases + 3 Glue jobs (bronze→silver, silver→gold, silver→ml-features)
- **Lambda** — 3 functions (ingestion, DQ check, ML inference)
- **EventBridge** — 3 cron rules for scheduled ingestion
- **Step Functions** — 2 state machines (ETL + ML)
- **SageMaker** — 2 Model Package Groups (forecast, anomaly)
- **IAM** — least-privilege roles for every service
- **SNS** — single alerts topic with optional email subscription

## Prerequisites

- Terraform `>= 1.5.0`
- AWS credentials configured (`aws configure` or `AWS_PROFILE`)
- Bucket + DynamoDB table for remote state (optional, see `versions.tf`)

## Usage

```bash
cd infra/terraform

# Copy the example and fill in your values
cp dev.example.tfvars dev.tfvars

# Supply the WAQI API token via env var (never commit it)
export TF_VAR_waqi_api_token="<your-waqi-token>"

# First-time setup
terraform init

# Preview changes
terraform plan -var-file=dev.tfvars

# Apply
terraform apply -var-file=dev.tfvars
```

## Upload Glue scripts to S3

Terraform creates the code bucket but does not upload scripts (this keeps the
IaC fast and deterministic). Upload them after the first `apply`:

```bash
CODE_BUCKET=$(terraform output -raw bucket_names | jq -r .code)

aws s3 sync ../../glue_jobs/ "s3://${CODE_BUCKET}/glue_jobs/" \
    --exclude "*" --include "*.py"
```

## Destroy

```bash
terraform destroy -var-file=dev.tfvars
```

Note: In **prod**, `force_destroy` on S3 buckets is `false`, so you must empty
them manually before `destroy`.

## Module structure

| File | Contents |
|------|----------|
| `versions.tf` | Required providers and backend config |
| `providers.tf` | AWS provider with default tags |
| `variables.tf` | Input variables |
| `locals.tf` | Naming prefixes and derived values |
| `s3.tf` | Medallion buckets + Athena results bucket |
| `glue.tf` | Glue catalog databases + 3 Glue jobs |
| `iam.tf` | Service-specific IAM roles (Glue, Lambda, SFN, SageMaker, Events) |
| `lambda.tf` | 3 Lambda functions + EventBridge schedules |
| `sns.tf` | Alerts SNS topic + optional email subscription |
| `step_functions.tf` | ETL + ML state machines (rendered via `templatefile`) |
| `sagemaker.tf` | Model Package Groups |
| `outputs.tf` | Bucket names, ARNs, function names |

## Cost estimate (dev)

Approximate monthly cost with minimal usage:

| Resource | Cost |
|----------|------|
| S3 (< 1 GB) | < \$0.10 |
| Lambda (3x/day ingestion + DQ) | < \$0.10 |
| Glue jobs (on-demand, G.1X, ~5 min/run) | \$0.30/run |
| Step Functions (< 1k executions) | \$0.03 |
| SageMaker (when training) | \$0.20/training job, \$0.10/batch transform |
| SNS (email) | free |
| **Total (idle + 30 runs)** | **≈ \$10–15 / month** |
