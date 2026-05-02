import json
import logging
import os
import sys
from datetime import datetime

from pathlib import Path

from helper import get_spark
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─── Configuration ──────────────────────────────────────────────────────────
LOCAL_MODEL_DIR = "/pipeline/data/models"
MODEL_PATH = f"file://{LOCAL_MODEL_DIR}/Gradient_Boosting"
HDFS_DATA_PATH = (
    "hdfs://namenode:9000/user/data-engineer/demand_forecasting/cleaned_data"
)
MANIFEST_OUT = "/pipeline/data/optimized_results/mlops_manifest.json"
MODEL_NAME = "Gradient_Boosting"
SAMPLE_ROWS = 100
FEATURE_COLS = ["Quantity", "Discount", "Tax", "ShippingCost", "Month", "DayOfWeek"]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("phase5_mlops")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    return logger


def load_model(
    local_model_dir: str, model_name: str, logger: logging.Logger
) -> PipelineModel:
    model_path = Path(local_model_dir) / model_name
    logger.info("[DEPLOY] Loading model from: %s", MODEL_PATH)
    if not model_path.exists():
        logger.error("Model path does not exist: %s", model_path)
        try:
            logger.error(
                "Available models in %s: %s",
                local_model_dir,
                sorted(os.listdir(local_model_dir)),
            )
        except OSError as exc:
            logger.error("Could not list model directory: %s", exc)
        raise RuntimeError("Model path missing")
    try:
        return PipelineModel.load(MODEL_PATH)
    except Exception as exc:
        logger.exception("Could not load Spark model from %s", MODEL_PATH)
        raise RuntimeError("Model load failed") from exc


def read_cleaned_data(spark: SparkSession, hdfs_path: str, logger: logging.Logger):
    logger.info("[AUTO] Reading cleaned data for inference...")
    return spark.read.parquet(hdfs_path)


def select_model_columns(df, logger: logging.Logger):
    df_features = (
        df.withColumn("OrderDate", F.to_date(F.col("OrderDate")))
        .withColumn("Quantity", F.col("Quantity").cast("double"))
        .withColumn("Discount", F.col("Discount").cast("double"))
        .withColumn("Tax", F.col("Tax").cast("double"))
        .withColumn("ShippingCost", F.col("ShippingCost").cast("double"))
        .withColumn("TotalAmount", F.col("TotalAmount").cast("double"))
        .withColumn("Month", F.month("OrderDate"))
        .withColumn("DayOfWeek", F.dayofweek("OrderDate"))
        .select("TotalAmount", *FEATURE_COLS)
    )

    logger.info("[AUTO] Feature columns prepared: %s", FEATURE_COLS)
    return df_features


def evaluate_predictions(predictions, logger: logging.Logger):
    logger.info("[MONITOR] Evaluating model accuracy...")
    evaluator = RegressionEvaluator(labelCol="TotalAmount", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    return rmse, r2


def write_manifest(
    path: str, rmse: float, r2: float, is_healthy: bool, logger: logging.Logger
) -> None:
    manifest = {
        "pipeline_run": datetime.now().isoformat(),
        "model_type": f"Spark_ML_{MODEL_NAME}",
        "deployment_source": "Local_Disk_Volume",
        "metrics": {"rmse": round(rmse, 4), "r2": round(r2, 4)},
        "monitoring": {
            "status": "HEALTHY" if is_healthy else "DEGRADED",
            "action_required": not is_healthy,
        },
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    logger.info("Manifest saved to: %s", path)


def main() -> int:
    logger = setup_logging()
    spark = get_spark("Phase5_MLOps_Local_Load")

    try:
        model = load_model(LOCAL_MODEL_DIR, MODEL_NAME, logger)
        df_cleaned = read_cleaned_data(spark, HDFS_DATA_PATH, logger)
        df_features = select_model_columns(df_cleaned, logger).limit(SAMPLE_ROWS)
        logger.info("[AUTO] Using sample size: %s", SAMPLE_ROWS)

        predictions = model.transform(df_features).cache()
        rmse, r2 = evaluate_predictions(predictions, logger)

        # Threshold logic: Flag if accuracy drops (RMSE increases)
        is_healthy = rmse < 1000.0
        write_manifest(MANIFEST_OUT, rmse, r2, is_healthy, logger)

        logger.info(
            "Phase 5 Complete. Model Health: %s", "[OK]" if is_healthy else "[ALERT]"
        )
        return 0
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
