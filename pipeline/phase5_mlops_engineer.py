import json
import logging
import os
import sys
from datetime import datetime

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─── Configuration ──────────────────────────────────────────────────────────
LOCAL_MODEL_DIR = "/pipeline/data/models"
HDFS_DATA_PATH = (
    "hdfs://namenode:9000/user/data-engineer/demand_forecasting/optimized_data"
)
MANIFEST_OUT = "/pipeline/data/optimized_results/mlops_manifest.json"
MODEL_NAME = "Random_Forest"
MODEL_VERSION = "v1"  # Placeholder for registry-based versioning.


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("phase5_mlops")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
    return logger


def get_spark(app_name: str) -> SparkSession:
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_model(
    local_model_dir: str, model_name: str, logger: logging.Logger
) -> PipelineModel:
    model_path = os.path.join(local_model_dir, model_name)
    logger.info("[DEPLOY] Loading model from: %s", model_path)
    try:
        return PipelineModel.load(model_path)
    except Exception as exc:
        logger.error("Could not load Spark model from %s", model_path)
        logger.error(
            "Ensure Phase 3 completed and the folder exists on your local device."
        )
        raise RuntimeError("Model load failed") from exc


def read_optimized_data(spark: SparkSession, hdfs_path: str, logger: logging.Logger):
    logger.info("[AUTO] Reading optimized data for inference...")
    return spark.read.parquet(hdfs_path)


def prepare_features(df_optimized, logger: logging.Logger):
    df_lower = df_optimized.toDF(*[c.lower() for c in df_optimized.columns])
    lower_cols = set(df_lower.columns)
    logger.info("[AUTO] Input columns: %s", sorted(lower_cols))

    order_date_col = "order_date" if "order_date" in lower_cols else "orderdate"
    if order_date_col not in lower_cols:
        raise ValueError("Missing required column: order_date/orderdate")
    if "quantity" not in lower_cols:
        raise ValueError("Missing required column: quantity")

    base_cols = {
        "OrderDate": F.to_date(F.col(order_date_col)),
        "Quantity": F.col("quantity").cast("double"),
        "Discount": (
            F.col("discount").cast("double") if "discount" in lower_cols else F.lit(0.0)
        ),
        "Tax": F.col("tax").cast("double") if "tax" in lower_cols else F.lit(0.0),
        "ShippingCost": (
            F.col("shippingcost").cast("double")
            if "shippingcost" in lower_cols
            else F.lit(0.0)
        ),
    }

    if "total_sales" in lower_cols:
        total_amount = F.col("total_sales").cast("double")
    elif "totalamount" in lower_cols:
        total_amount = F.col("totalamount").cast("double")
    elif "price" in lower_cols and "quantity" in lower_cols:
        total_amount = (F.col("price") * F.col("quantity")).cast("double")
    else:
        raise ValueError(
            "Missing TotalAmount; expected total_sales, totalamount, or price+quantity"
        )

    df_features = (
        df_lower.withColumn("OrderDate", base_cols["OrderDate"])
        .withColumn("Quantity", base_cols["Quantity"])
        .withColumn("Discount", base_cols["Discount"])
        .withColumn("Tax", base_cols["Tax"])
        .withColumn("ShippingCost", base_cols["ShippingCost"])
        .withColumn("TotalAmount", total_amount)
        .withColumn("Month", F.month("OrderDate"))
        .withColumn("DayOfWeek", F.dayofweek("OrderDate"))
        .select(
            "OrderDate",
            "Quantity",
            "Discount",
            "Tax",
            "ShippingCost",
            "TotalAmount",
            "Month",
            "DayOfWeek",
        )
    )

    logger.info("[AUTO] Feature columns prepared")
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
        "model_type": "Spark_ML_Random_Forest",
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
        df_optimized = read_optimized_data(spark, HDFS_DATA_PATH, logger)
        df_features = prepare_features(df_optimized, logger)

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
