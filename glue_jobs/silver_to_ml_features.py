"""
Glue Job: Silver fact_aqi → ML Features
─────────────────────────────────────────
Build features for ML (forecasting + anomaly detection).
Output partitioned Parquet ready for SageMaker training / batch transform.

Features produced per (queried_city, measured_at):
  - Lag features : aqi_lag_{1,3,6,12,24}h, pm25_lag_{1,3,6,12,24}h
  - Rolling     : aqi_roll_{mean,std,min,max}_{3h,24h,7d}
  - Rate        : aqi_diff_1h, aqi_pct_change_24h
  - Time        : hour, dow, month, is_weekend, is_dry_season
                  hour_sin, hour_cos, month_sin, month_cos
  - Target      : aqi_target (t + forecast_horizon_h)

Job Parameters:
    --JOB_NAME
    --silver_database      e.g. glue-pipeline-silver-dev
    --silver_table         default: fact_aqi
    --ml_features_bucket   e.g. data-pipeline-ml-ap-dev
    --ml_features_database e.g. glue-pipeline-ml-dev
    --forecast_horizon_h   default: 24
    --min_measured_at      optional ISO date, filter input start (e.g. 2021-01-01)
"""

import math
import sys

from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window


# ── Setup ─────────────────────────────────────────────────────────────────────
REQUIRED_ARGS = [
    "JOB_NAME",
    "silver_database",
    "ml_features_bucket",
    "ml_features_database",
]
OPTIONAL_ARGS = {
    "silver_table": "fact_aqi",
    "forecast_horizon_h": "24",
    "min_measured_at": "",
}

args = getResolvedOptions(sys.argv, REQUIRED_ARGS + list(OPTIONAL_ARGS.keys()))
for k, default in OPTIONAL_ARGS.items():
    if not args.get(k):
        args[k] = default

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)
logger = glueContext.get_logger()

SILVER_DB = args["silver_database"]
SILVER_TABLE = args["silver_table"]
ML_BUCKET = args["ml_features_bucket"]
ML_DB = args["ml_features_database"]
HORIZON_H = int(args["forecast_horizon_h"])
MIN_TS = args["min_measured_at"]

FEATURES_PATH = f"s3://{ML_BUCKET}/features/aqi_features/"
FEATURES_TABLE = "aqi_features"

logger.info(f"Silver: {SILVER_DB}.{SILVER_TABLE}")
logger.info(f"Output: {FEATURES_PATH}  (catalog: {ML_DB}.{FEATURES_TABLE})")
logger.info(f"Forecast horizon: {HORIZON_H}h")


# ── Step 1: Read Silver fact_aqi ──────────────────────────────────────────────
logger.info("Reading Silver fact_aqi...")
src = glueContext.create_dynamic_frame.from_catalog(
    database=SILVER_DB,
    table_name=SILVER_TABLE,
    transformation_ctx="silver_fact_aqi",
)
df = src.toDF()
raw_count = df.count()
logger.info(f"Silver rows: {raw_count:,}")

if raw_count == 0:
    logger.info("Empty Silver table. Committing.")
    job.commit()
    sys.exit(0)

if MIN_TS:
    df = df.filter(F.col("measured_at") >= F.lit(MIN_TS).cast("timestamp"))
    logger.info(f"After min_measured_at filter: {df.count():,} rows")


# ── Step 2: Hourly aggregation per city ───────────────────────────────────────
logger.info("Aggregating to hourly granularity per city...")

df_h = (
    df.withColumn("hour_ts", F.date_trunc("hour", F.col("measured_at")))
      .groupBy("queried_city", "hour_ts")
      .agg(
          F.avg("aqi").alias("aqi"),
          F.avg("pm25").alias("pm25"),
          F.avg("pm10").alias("pm10"),
          F.avg("temperature").alias("temperature"),
          F.avg("humidity").alias("humidity"),
          F.avg("pressure").alias("pressure"),
          F.avg("wind").alias("wind"),
      )
      .withColumnRenamed("hour_ts", "measured_at")
      .filter(F.col("aqi").isNotNull())
)
logger.info(f"Hourly rows: {df_h.count():,}")


# ── Step 3: Lag + rolling features (per city, ordered by time) ────────────────
logger.info("Computing lag + rolling features...")

w = Window.partitionBy("queried_city").orderBy("measured_at")

def rolling_window(hours: int) -> Window:
    # rowsBetween is in rows (hourly granularity → rows ≈ hours)
    # shift by 1 so "rolling" excludes current row (avoid target leakage)
    return (
        Window.partitionBy("queried_city")
        .orderBy("measured_at")
        .rowsBetween(-hours, -1)
    )

for lag in [1, 3, 6, 12, 24]:
    df_h = df_h.withColumn(f"aqi_lag_{lag}h", F.lag("aqi", lag).over(w))
    df_h = df_h.withColumn(f"pm25_lag_{lag}h", F.lag("pm25", lag).over(w))

for hours, label in [(3, "3h"), (24, "24h"), (24 * 7, "7d")]:
    rw = rolling_window(hours)
    df_h = df_h.withColumn(f"aqi_roll_mean_{label}", F.avg("aqi").over(rw))
    df_h = df_h.withColumn(f"aqi_roll_std_{label}", F.stddev("aqi").over(rw))
    df_h = df_h.withColumn(f"aqi_roll_min_{label}", F.min("aqi").over(rw))
    df_h = df_h.withColumn(f"aqi_roll_max_{label}", F.max("aqi").over(rw))

df_h = df_h.withColumn("aqi_diff_1h", F.col("aqi") - F.lag("aqi", 1).over(w))
df_h = df_h.withColumn(
    "aqi_pct_change_24h",
    (F.col("aqi") - F.lag("aqi", 24).over(w)) /
    F.when(F.lag("aqi", 24).over(w) == 0, F.lit(None)).otherwise(F.lag("aqi", 24).over(w)),
)


# ── Step 4: Time features ─────────────────────────────────────────────────────
logger.info("Adding time features...")

df_h = df_h.withColumn("hour",  F.hour("measured_at"))
df_h = df_h.withColumn("dow",   F.dayofweek("measured_at"))
df_h = df_h.withColumn("month", F.month("measured_at"))
df_h = df_h.withColumn("is_weekend", F.when(F.col("dow").isin(1, 7), 1).otherwise(0))
df_h = df_h.withColumn(
    "is_dry_season",
    F.when(F.col("month").isin(11, 12, 1, 2, 3, 4), 1).otherwise(0),
)

two_pi = F.lit(2 * math.pi)
df_h = df_h.withColumn("hour_sin",  F.sin(two_pi * F.col("hour")  / F.lit(24.0)))
df_h = df_h.withColumn("hour_cos",  F.cos(two_pi * F.col("hour")  / F.lit(24.0)))
df_h = df_h.withColumn("month_sin", F.sin(two_pi * F.col("month") / F.lit(12.0)))
df_h = df_h.withColumn("month_cos", F.cos(two_pi * F.col("month") / F.lit(12.0)))


# ── Step 5: Target — AQI at t + HORIZON_H ─────────────────────────────────────
logger.info(f"Adding target aqi_target = aqi(t + {HORIZON_H}h)...")

w_fwd = Window.partitionBy("queried_city").orderBy("measured_at")
df_h = df_h.withColumn("aqi_target", F.lead("aqi", HORIZON_H).over(w_fwd))


# ── Step 6: Partition columns ─────────────────────────────────────────────────
df_h = df_h.withColumn("year",  F.year("measured_at").cast(StringType()))
df_h = df_h.withColumn(
    "month_part",
    F.lpad(F.month("measured_at").cast(StringType()), 2, "0"),
)


# ── Step 7: Write to S3 + update Glue Catalog ─────────────────────────────────
logger.info(f"Writing features to {FEATURES_PATH} ...")

out_dynf = DynamicFrame.fromDF(df_h, glueContext, "aqi_features")
sink = glueContext.getSink(
    connection_type="s3",
    path=FEATURES_PATH,
    enableUpdateCatalog=True,
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=["queried_city", "year", "month_part"],
)
sink.setCatalogInfo(catalogDatabase=ML_DB, catalogTableName=FEATURES_TABLE)
sink.setFormat("glueparquet", compression="snappy")
sink.writeFrame(out_dynf)

logger.info(f"ML features written: {df_h.count():,} rows")
logger.info("silver_to_ml_features job complete.")

job.commit()
