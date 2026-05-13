"""
SageMaker Training Script — AQI Anomaly Detection (Isolation Forest)
─────────────────────────────────────────────────────────────────────
Unsupervised anomaly detection using Isolation Forest on multivariate
air quality + weather features.

Artifacts saved to SM_MODEL_DIR:
    - iforest_model.joblib
    - scaler.joblib
    - feature_columns.json
    - metadata.json (contamination, threshold, feature names)
    - model.tar.gz (ready for SageMaker Model Registry)

Run-time metrics emitted to CloudWatch:
    - training:anomaly_rate=...
    - training:score_threshold=...
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("aqi-anomaly-train")


DEFAULT_FEATURES = [
    "aqi", "pm25", "pm10",
    "temperature", "humidity", "pressure", "wind",
]


def read_features(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    logger.info(f"Reading {len(files)} parquet files from {data_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "measured_at" in df.columns:
        df["measured_at"] = pd.to_datetime(df["measured_at"], utc=True)
    return df


def maybe_register_model(
    model_path: str,
    metadata: dict,
    model_package_group: str,
    region: str,
) -> str | None:
    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not available — skipping Model Registry registration")
        return None

    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    try:
        sm.describe_model_package_group(ModelPackageGroupName=model_package_group)
    except sm.exceptions.ClientError:
        logger.info(f"Creating ModelPackageGroup: {model_package_group}")
        sm.create_model_package_group(
            ModelPackageGroupName=model_package_group,
            ModelPackageGroupDescription="Vietnam Air Quality anomaly detectors",
        )

    artifact_bucket = os.environ.get("SM_TRAINING_BUCKET")
    if not artifact_bucket:
        logger.warning("SM_TRAINING_BUCKET not set — registration skipped")
        return None

    key = f"ml-artifacts/aqi-anomaly/model-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.tar.gz"
    s3.upload_file(model_path, artifact_bucket, key)
    model_url = f"s3://{artifact_bucket}/{key}"
    logger.info(f"Uploaded model artifact → {model_url}")

    import sagemaker
    sklearn_image = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        image_scope="inference",
        instance_type="ml.m5.xlarge",
    )

    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group,
        ModelPackageDescription=(
            f"IsolationForest — contamination={metadata['contamination']}, "
            f"anomaly_rate_observed={metadata['anomaly_rate_observed']:.4f}"
        ),
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{"Image": sklearn_image, "ModelDataUrl": model_url}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
        },
        CustomerMetadataProperties={
            "contamination": str(metadata["contamination"]),
            "anomaly_score_threshold": f"{metadata['anomaly_score_threshold']:.6f}",
            "anomaly_rate_observed": f"{metadata['anomaly_rate_observed']:.6f}",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    arn = response["ModelPackageArn"]
    logger.info(f"Registered Model Package: {arn}")
    return arn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contamination", type=float, default=0.02)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_samples", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--feature_cols", type=str, default=",".join(DEFAULT_FEATURES),
                   help="Comma-separated list of input feature column names")

    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

    p.add_argument("--model_package_group", type=str, default="")
    p.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", "ap-southeast-2"))

    args = p.parse_args()
    np.random.seed(args.seed)
    logger.info(f"Args: {vars(args)}")

    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    logger.info(f"Using features: {feature_cols}")

    # ── Load + clean ─────────────────────────────────────────────────────────
    df = read_features(args.train)
    logger.info(f"Total rows: {len(df):,}")

    df = df.dropna(subset=feature_cols).copy()
    logger.info(f"After dropna on features: {len(df):,}")
    if len(df) < 100:
        raise RuntimeError("Too few rows after dropna — cannot train.")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("Training IsolationForest...")
    ms: str | int
    try:
        ms = int(args.max_samples)
    except ValueError:
        ms = args.max_samples  # "auto"

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=ms,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X)

    pred = model.predict(X)
    is_anom = (pred == -1).astype(int)
    score = -model.score_samples(X)

    anomaly_rate = float(is_anom.mean())
    threshold = float(score[is_anom == 1].min()) if is_anom.sum() > 0 else float(score.max())

    logger.info(f"Anomaly rate : {anomaly_rate:.4f}")
    logger.info(f"Score threshold: {threshold:.6f}")

    print(f"training:anomaly_rate={anomaly_rate:.6f};")
    print(f"training:score_threshold={threshold:.6f};")
    print(f"training:n_rows={len(df)};")

    # ── Save artifacts ───────────────────────────────────────────────────────
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "iforest_model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")

    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    meta = {
        "model_type": "isolation_forest",
        "feature_cols": feature_cols,
        "contamination": args.contamination,
        "n_estimators": args.n_estimators,
        "anomaly_score_threshold": threshold,
        "anomaly_rate_observed": anomaly_rate,
        "n_rows_trained_on": int(len(df)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    tar_path = model_dir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in model_dir.iterdir():
            if f.name != "model.tar.gz":
                tar.add(f, arcname=f.name)
    logger.info(f"Model saved → {tar_path}")

    # ── Register ─────────────────────────────────────────────────────────────
    if args.model_package_group:
        try:
            arn = maybe_register_model(
                model_path=str(tar_path),
                metadata=meta,
                model_package_group=args.model_package_group,
                region=args.region,
            )
            if arn:
                with open(model_dir / "model_package_arn.txt", "w") as f:
                    f.write(arn)
        except Exception as e:
            logger.error(f"Model registration failed: {e}")

    logger.info("Training done.")


if __name__ == "__main__":
    main()
