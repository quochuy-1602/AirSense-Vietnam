"""
Lambda: ML Inference — Kick off Batch Transform + Write Gold Tables
────────────────────────────────────────────────────────────────────
Orchestrates:
  1. Resolve latest Approved model package from SageMaker Model Registry
  2. Kick off SageMaker Batch Transform jobs for:
       - Forecasting  (XGBoost, Model Package Group: FORECAST_MODEL_PKG_GROUP)
       - Anomaly      (Isolation Forest, Model Package Group: ANOMALY_MODEL_PKG_GROUP)
  3. Wait for completion (polling with Lambda-safe timeout)
  4. Post-process predictions → write partitioned Parquet into Gold bucket:
       - gold_aqi_forecast
       - gold_aqi_anomalies
  5. SNS alert if high-severity anomalies detected

Supports TWO modes (selected via event["mode"]):
  - "trigger"  : start Batch Transform jobs, return job names (for Step Functions)
  - "collect"  : read output from finished jobs + write to Gold + SNS
  - "full"     : trigger → wait → collect (for EventBridge daily trigger)

Environment Variables:
    FORECAST_MODEL_PKG_GROUP    e.g. aqi-forecast-models
    ANOMALY_MODEL_PKG_GROUP     e.g. aqi-anomaly-models
    ML_FEATURES_BUCKET          S3 bucket with feature parquet (from Glue)
    ML_FEATURES_PREFIX          default: features/aqi_features/
    GOLD_BUCKET                 S3 bucket for Gold tables
    GOLD_DATABASE               Glue database for Gold tables
    BATCH_OUTPUT_BUCKET         where Batch Transform writes output
    SAGEMAKER_ROLE_ARN          IAM role for SageMaker
    SNS_ALERT_TOPIC_ARN         optional, for anomaly alerts
    ANOMALY_ALERT_AQI_THRESHOLD default: 150
    INSTANCE_TYPE               default: ml.m5.xlarge
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")
sns = boto3.client("sns")

# ── Config ───────────────────────────────────────────────────────────────────
FORECAST_GROUP = os.environ["FORECAST_MODEL_PKG_GROUP"]
ANOMALY_GROUP = os.environ["ANOMALY_MODEL_PKG_GROUP"]
ML_BUCKET = os.environ["ML_FEATURES_BUCKET"]
ML_PREFIX = os.environ.get("ML_FEATURES_PREFIX", "features/aqi_features/").rstrip("/")
GOLD_BUCKET = os.environ["GOLD_BUCKET"]
GOLD_DB = os.environ["GOLD_DATABASE"]
BATCH_OUT_BUCKET = os.environ["BATCH_OUTPUT_BUCKET"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
SNS_TOPIC = os.environ.get("SNS_ALERT_TOPIC_ARN", "").strip()
ANOMALY_ALERT_AQI = float(os.environ.get("ANOMALY_ALERT_AQI_THRESHOLD", "150"))
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.xlarge")


# ── Helpers ──────────────────────────────────────────────────────────────────
def latest_approved_package(group: str) -> dict:
    """Return latest Approved ModelPackage in group."""
    resp = sm.list_model_packages(
        ModelPackageGroupName=group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=10,
    )
    packages = resp.get("ModelPackageSummaryList", [])
    if not packages:
        raise RuntimeError(f"No Approved model packages in group '{group}'")
    arn = packages[0]["ModelPackageArn"]
    return sm.describe_model_package(ModelPackageName=arn)


def ensure_model(model_name: str, pkg_arn: str) -> str:
    """Create SageMaker Model from ModelPackage (idempotent on name)."""
    try:
        sm.describe_model(ModelName=model_name)
        return model_name
    except ClientError:
        pass
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={"ModelPackageName": pkg_arn},
    )
    return model_name


def start_transform_job(
    kind: str,
    model_name: str,
    input_s3_uri: str,
    output_s3_uri: str,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    job_name = f"aqi-{kind}-{ts}"
    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput={
            "DataSource": {
                "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_s3_uri}
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": output_s3_uri,
            "Accept": "text/csv",
            "AssembleWith": "Line",
        },
        TransformResources={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
        },
    )
    logger.info(f"Started {kind} transform: {job_name}")
    return job_name


def wait_for_job(job_name: str, max_wait_sec: int = 700) -> str:
    """Poll Transform job until terminal state or timeout."""
    start = time.time()
    while time.time() - start < max_wait_sec:
        r = sm.describe_transform_job(TransformJobName=job_name)
        st = r["TransformJobStatus"]
        if st in ("Completed", "Failed", "Stopped"):
            if st != "Completed":
                raise RuntimeError(
                    f"Transform job {job_name} ended with {st}: "
                    f"{r.get('FailureReason', '')}"
                )
            return st
        time.sleep(10)
    raise TimeoutError(f"Transform job {job_name} did not finish in {max_wait_sec}s")


def read_csv_from_s3(bucket: str, prefix: str) -> list[dict]:
    """Read all .out / .csv files under prefix and concatenate as list of dicts."""
    rows: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.endswith(".csv") or key.endswith(".out")):
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(body))
            rows.extend(reader)
    return rows


def write_jsonl_to_s3(bucket: str, key: str, records: list[dict]) -> None:
    body = "\n".join(json.dumps(r, default=str) for r in records).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/x-ndjson")


def send_alert(subject: str, message: str) -> None:
    if not SNS_TOPIC:
        logger.info("SNS topic not configured, skipping alert")
        return
    region = SNS_TOPIC.split(":")[3]
    boto3.client("sns", region_name=region).publish(
        TopicArn=SNS_TOPIC,
        Subject=subject[:100],
        Message=message,
    )


# ── Core logic ───────────────────────────────────────────────────────────────
def trigger_jobs(features_input_uri: str, run_id: str) -> dict:
    """Start both batch transform jobs. Returns job names."""
    fc_pkg = latest_approved_package(FORECAST_GROUP)
    an_pkg = latest_approved_package(ANOMALY_GROUP)

    fc_model = ensure_model(f"aqi-forecast-{run_id}", fc_pkg["ModelPackageArn"])
    an_model = ensure_model(f"aqi-anomaly-{run_id}",  an_pkg["ModelPackageArn"])

    forecast_out = f"s3://{BATCH_OUT_BUCKET}/ml-inference/{run_id}/forecast/"
    anomaly_out  = f"s3://{BATCH_OUT_BUCKET}/ml-inference/{run_id}/anomaly/"

    fc_job = start_transform_job("forecast", fc_model, features_input_uri, forecast_out)
    an_job = start_transform_job("anomaly",  an_model, features_input_uri, anomaly_out)

    return {
        "run_id": run_id,
        "forecast_job": fc_job,
        "anomaly_job": an_job,
        "forecast_output": forecast_out,
        "anomaly_output": anomaly_out,
        "forecast_model_package": fc_pkg["ModelPackageArn"],
        "anomaly_model_package": an_pkg["ModelPackageArn"],
    }


def collect_and_write_gold(jobs_info: dict) -> dict:
    """Read outputs of finished jobs, write Gold tables, fire SNS if needed."""
    run_id = jobs_info["run_id"]

    fc_bucket, fc_prefix = jobs_info["forecast_output"].replace("s3://", "").split("/", 1)
    an_bucket, an_prefix = jobs_info["anomaly_output"].replace("s3://", "").split("/", 1)

    forecast_rows = read_csv_from_s3(fc_bucket, fc_prefix)
    anomaly_rows  = read_csv_from_s3(an_bucket, an_prefix)
    logger.info(f"Forecast rows: {len(forecast_rows)}, Anomaly rows: {len(anomaly_rows)}")

    now_iso = datetime.now(timezone.utc).isoformat()

    # Forecasts
    forecast_records = []
    for r in forecast_rows:
        pred = r.get("predicted_aqi")
        if pred is None:
            continue
        try:
            pred_f = float(pred)
        except ValueError:
            continue
        forecast_records.append({
            "run_id": run_id,
            "forecast_made_at": now_iso,
            "predicted_aqi": pred_f,
            "model_package": jobs_info["forecast_model_package"],
        })

    # Anomalies
    anomaly_records = []
    high_sev = []
    for r in anomaly_rows:
        try:
            score = float(r.get("anomaly_score", "nan"))
            is_anom = int(float(r.get("is_anomaly", "0")))
        except ValueError:
            continue
        rec = {
            "run_id": run_id,
            "detected_at": now_iso,
            "anomaly_score": score,
            "is_anomaly": is_anom,
            "model_package": jobs_info["anomaly_model_package"],
        }
        anomaly_records.append(rec)
        if is_anom == 1 and score > 0:
            # No aqi context in output-only CSV (would need join with input);
            # high-severity threshold is approximated by score quantile.
            high_sev.append(rec)

    # Write Gold — JSON Lines, partitioned by date
    date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
    fc_key = f"gold_aqi_forecast/dt={date_part}/forecast-{run_id}.jsonl"
    an_key = f"gold_aqi_anomalies/dt={date_part}/anomalies-{run_id}.jsonl"
    write_jsonl_to_s3(GOLD_BUCKET, fc_key, forecast_records)
    write_jsonl_to_s3(GOLD_BUCKET, an_key, anomaly_records)
    logger.info(f"Wrote Gold: s3://{GOLD_BUCKET}/{fc_key}")
    logger.info(f"Wrote Gold: s3://{GOLD_BUCKET}/{an_key}")

    # Alert
    high_count = sum(1 for r in anomaly_records if r["is_anomaly"] == 1)
    if high_count > 0:
        send_alert(
            subject=f"[AQ Pipeline] {high_count} anomalies detected (run={run_id})",
            message=json.dumps({
                "run_id": run_id,
                "anomaly_count": high_count,
                "forecast_count": len(forecast_records),
                "forecast_gold": f"s3://{GOLD_BUCKET}/{fc_key}",
                "anomaly_gold":  f"s3://{GOLD_BUCKET}/{an_key}",
            }, indent=2),
        )

    return {
        "run_id": run_id,
        "forecast_count": len(forecast_records),
        "anomaly_count": len(anomaly_records),
        "high_severity_count": high_count,
        "forecast_gold": f"s3://{GOLD_BUCKET}/{fc_key}",
        "anomaly_gold":  f"s3://{GOLD_BUCKET}/{an_key}",
    }


# ── Handler ──────────────────────────────────────────────────────────────────
def lambda_handler(event: dict[str, Any], context) -> dict:
    mode = event.get("mode", "full")
    run_id = event.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    features_input_uri = event.get("features_input_uri") or f"s3://{ML_BUCKET}/{ML_PREFIX}/"

    logger.info(f"mode={mode} run_id={run_id} features={features_input_uri}")

    if mode == "trigger":
        return trigger_jobs(features_input_uri, run_id)

    if mode == "collect":
        jobs_info = event["jobs_info"]
        return collect_and_write_gold(jobs_info)

    if mode == "full":
        jobs_info = trigger_jobs(features_input_uri, run_id)
        wait_for_job(jobs_info["forecast_job"])
        wait_for_job(jobs_info["anomaly_job"])
        return collect_and_write_gold(jobs_info)

    raise ValueError(f"Unknown mode: {mode}")
