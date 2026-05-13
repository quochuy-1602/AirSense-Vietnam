"""
SageMaker Training Script — AQI 24h Forecast (XGBoost)
───────────────────────────────────────────────────────
Designed to run in SageMaker Training Job on a script-mode image.

SageMaker conventions used:
  - Input features path     : SM_CHANNEL_TRAIN (default /opt/ml/input/data/train)
  - Model output directory  : SM_MODEL_DIR     (default /opt/ml/model)
  - Output artifacts dir    : SM_OUTPUT_DATA_DIR
  - Hyperparameters passed as CLI args

Model Registry:
  When --model_package_group is provided, the trained model is registered as
  a new ModelPackage (PendingManualApproval by default, or Approved if
  --auto_approve and test MAE improves vs baseline).

Usage (SageMaker Python SDK — see README section):
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="sagemaker/forecasting",
        framework_version="1.2-1",
        instance_type="ml.m5.xlarge",
        hyperparameters={
            "forecast_horizon_h": 24,
            "max_depth": 6,
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "model_package_group": "aqi-forecast-models",
            "auto_approve_mae_threshold": 20.0,
        },
    )
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("aqi-forecast-train")


# ── Helpers ──────────────────────────────────────────────────────────────────
def read_features(data_dir: str) -> pd.DataFrame:
    """Read all Parquet files from SageMaker channel directory."""
    data_dir = Path(data_dir)
    files = sorted(glob.glob(str(data_dir / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    logger.info(f"Reading {len(files)} parquet files from {data_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["measured_at"] = pd.to_datetime(df["measured_at"], utc=True)
    return df


def make_time_split(
    df: pd.DataFrame, train_end: str, val_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_end_ts = pd.Timestamp(val_end, tz="UTC")
    tr = df[df["measured_at"] < train_end_ts]
    va = df[(df["measured_at"] >= train_end_ts) & (df["measured_at"] < val_end_ts)]
    te = df[df["measured_at"] >= val_end_ts]
    return tr, va, te


def evaluate(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape, "n": int(len(y_true))}


# ── Model Registry helper ────────────────────────────────────────────────────
def maybe_register_model(
    model_path: str,
    metrics: dict,
    model_package_group: str,
    auto_approve_mae_threshold: float | None,
    region: str,
) -> str | None:
    """Upload model artifact + register in SageMaker Model Registry."""
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
            ModelPackageGroupDescription="Vietnam Air Quality forecast models (XGBoost 24h)",
        )

    artifact_bucket = os.environ.get("SM_TRAINING_BUCKET")
    artifact_key = f"ml-artifacts/aqi-forecast/model-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.tar.gz"
    if artifact_bucket:
        s3.upload_file(model_path, artifact_bucket, artifact_key)
        model_url = f"s3://{artifact_bucket}/{artifact_key}"
        logger.info(f"Uploaded model artifact → {model_url}")
    else:
        logger.warning(
            "SM_TRAINING_BUCKET not set — using SM_MODEL_DIR path (local) which won't be resolvable."
        )
        model_url = f"file://{model_path}"

    approval = "PendingManualApproval"
    if auto_approve_mae_threshold is not None and metrics["mae"] <= auto_approve_mae_threshold:
        approval = "Approved"
        logger.info(f"Auto-approving: MAE={metrics['mae']:.3f} <= {auto_approve_mae_threshold}")

    import sagemaker
    xgb_image = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        image_scope="inference",
        instance_type="ml.m5.xlarge",
    )

    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group,
        ModelPackageDescription=(
            f"AQI 24h XGBoost — MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f}"
        ),
        ModelApprovalStatus=approval,
        InferenceSpecification={
            "Containers": [{"Image": xgb_image, "ModelDataUrl": model_url}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": model_url.replace(".tar.gz", "_metrics.json"),
                }
            }
        },
        CustomerMetadataProperties={
            "mae": f"{metrics['mae']:.4f}",
            "rmse": f"{metrics['rmse']:.4f}",
            "mape": f"{metrics['mape']:.4f}",
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    arn = response["ModelPackageArn"]
    logger.info(f"Registered Model Package: {arn} (status={approval})")
    return arn


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--forecast_horizon_h", type=int, default=24)
    p.add_argument("--train_end", type=str, default="2021-10-01")
    p.add_argument("--val_end", type=str, default="2021-11-01")

    p.add_argument("--n_estimators", type=int, default=1000)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--min_child_weight", type=int, default=5)
    p.add_argument("--reg_alpha", type=float, default=0.1)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    p.add_argument("--model_package_group", type=str, default="")
    p.add_argument("--auto_approve_mae_threshold", type=float, default=float("nan"))
    p.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", "ap-southeast-2"))

    args = p.parse_args()
    np.random.seed(args.seed)

    logger.info(f"Args: {vars(args)}")

    # ── Load ─────────────────────────────────────────────────────────────────
    df = read_features(args.train)
    logger.info(f"Total rows: {len(df):,} | date range {df['measured_at'].min()} → {df['measured_at'].max()}")

    df = df.dropna(subset=["aqi_target"]).copy()
    if "queried_city" in df.columns:
        cat = pd.Categorical(df["queried_city"])
        df["city_code"] = cat.codes
        city_mapping = dict(enumerate(cat.categories))
    else:
        df["city_code"] = 0
        city_mapping = {0: "unknown"}

    drop_cols = {"measured_at", "queried_city", "aqi_target",
                 "year", "month_part"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    df = df.dropna(subset=feature_cols)
    logger.info(f"After dropna: {len(df):,} rows, {len(feature_cols)} features")

    # ── Split ────────────────────────────────────────────────────────────────
    tr, va, te = make_time_split(df, args.train_end, args.val_end)
    logger.info(f"Train: {len(tr):,} | Val: {len(va):,} | Test: {len(te):,}")
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        raise RuntimeError("One of the splits is empty — check train_end / val_end dates vs input data.")

    X_tr, y_tr = tr[feature_cols], tr["aqi_target"]
    X_va, y_va = va[feature_cols], va["aqi_target"]
    X_te, y_te = te[feature_cols], te["aqi_target"]

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        tree_method="hist",
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=-1,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
    logger.info(f"Best iteration: {xgb.best_iteration}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    val_metrics = evaluate(y_va, xgb.predict(X_va))
    test_metrics = evaluate(y_te, xgb.predict(X_te))
    logger.info(f"VAL:  {val_metrics}")
    logger.info(f"TEST: {test_metrics}")

    # SageMaker metric emission — picked up by CloudWatch
    print(f"validation:mae={val_metrics['mae']:.4f};")
    print(f"validation:rmse={val_metrics['rmse']:.4f};")
    print(f"test:mae={test_metrics['mae']:.4f};")
    print(f"test:rmse={test_metrics['rmse']:.4f};")
    print(f"test:mape={test_metrics['mape']:.4f};")

    # ── SHAP Feature Importance ───────────────────────────────────────────────
    logger.info("Computing SHAP feature importance on test set sample…")
    try:
        import shap as _shap
        _sample_n = min(500, len(X_te))
        _explainer = _shap.TreeExplainer(xgb)
        _shap_vals = _explainer.shap_values(X_te.iloc[:_sample_n])
        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(_shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(model_dir / "shap_importance.csv", index=False)
        np.save(str(model_dir / "shap_values_sample.npy"), _shap_vals[:min(100, _sample_n)])
        logger.info(f"Top 5 SHAP features:\n{shap_importance.head().to_string(index=False)}")
    except Exception as _e:
        logger.warning(f"SHAP skipped (non-critical): {_e}")

    # ── Save model ───────────────────────────────────────────────────────────
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb, model_dir / "xgb_model.joblib")
    with open(model_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(model_dir / "city_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in city_mapping.items()}, f, indent=2)
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "model_type": "xgboost_regressor",
                "forecast_horizon_h": args.forecast_horizon_h,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "best_iteration": int(xgb.best_iteration or 0),
                "n_features": len(feature_cols),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "hyperparameters": {
                    k: v for k, v in vars(args).items()
                    if k not in ("train", "model_dir", "output_data_dir")
                },
            },
            f,
            indent=2,
            default=str,
        )

    # Pack into .tar.gz for Model Registry
    tar_path = model_dir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in model_dir.iterdir():
            if f.name != "model.tar.gz":
                tar.add(f, arcname=f.name)
    logger.info(f"Model saved → {tar_path}")

    # ── Optional: Register in Model Registry ─────────────────────────────────
    if args.model_package_group:
        auto_threshold = args.auto_approve_mae_threshold if not (
            isinstance(args.auto_approve_mae_threshold, float) and
            np.isnan(args.auto_approve_mae_threshold)
        ) else None
        try:
            arn = maybe_register_model(
                model_path=str(tar_path),
                metrics=test_metrics,
                model_package_group=args.model_package_group,
                auto_approve_mae_threshold=auto_threshold,
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
