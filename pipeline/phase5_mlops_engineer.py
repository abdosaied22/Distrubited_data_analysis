import os
import json
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

# ─── 1. Spark Setup ──────────────────────────────────────────────────────────
spark = SparkSession.builder.appName("Phase5_MLOps_Local_Load").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Use the LOCAL directory defined in your Phase 3
LOCAL_MODEL_DIR = "/pipeline/data/models"
# Use the HDFS path where Big Data Engineer (Phase 4) saved the optimized data
HDFS_DATA_PATH = "hdfs://namenode:9000/user/data-engineer/demand_forecasting/optimized_data"
# Path to save the health report
MANIFEST_OUT = "/pipeline/data/optimized_results/mlops_manifest.json"

# ─── 2. DEPLOYMENT: Load Spark Model from Local Disk ───────────────────────
# We use Random_Forest as the "Production" model
model_path = os.path.join(LOCAL_MODEL_DIR, "Random_Forest")

print(f"\n[DEPLOY] Loading model from: {model_path}")

try:
    # IMPORTANT: Loading from the FOLDER structure, not a .pkl
    model = PipelineModel.load(model_path)
    print("✅ Model successfully deployed into production.")
except Exception as e:
    print(f"❌ ERROR: Could not load Spark model from {model_path}")
    print("Ensure Phase 3 completed and the folder exists on your local device.")
    spark.stop()
    sys.exit(1)

# ─── 3. AUTOMATION: Run Inference ──────────────────────────────────────────
print(f"[AUTO] Reading optimized data for inference...")
try:
    df_optimized = spark.read.parquet(HDFS_DATA_PATH)
    df_lower = df_optimized.toDF(*[c.lower() for c in df_optimized.columns])
    lower_cols = set(df_lower.columns)
    if "order_date" not in lower_cols:
        raise ValueError("Missing required column: order_date")

    df_features = df_lower.withColumn("OrderDate", F.to_date(F.col("order_date")))

    if "quantity" in lower_cols:
        df_features = df_features.withColumn(
            "Quantity", F.col("quantity").cast("double")
        )
    else:
        raise ValueError("Missing required column: quantity")

    if "discount" in lower_cols:
        df_features = df_features.withColumn(
            "Discount", F.col("discount").cast("double")
        )
    else:
        df_features = df_features.withColumn("Discount", F.lit(0.0))

    if "tax" in lower_cols:
        df_features = df_features.withColumn("Tax", F.col("tax").cast("double"))
    else:
        df_features = df_features.withColumn("Tax", F.lit(0.0))

    if "shippingcost" in lower_cols:
        df_features = df_features.withColumn(
            "ShippingCost", F.col("shippingcost").cast("double")
        )
    else:
        df_features = df_features.withColumn("ShippingCost", F.lit(0.0))

    if "total_sales" in lower_cols:
        df_features = df_features.withColumn(
            "TotalAmount", F.col("total_sales").cast("double")
        )
    elif "totalamount" in lower_cols:
        df_features = df_features.withColumn(
            "TotalAmount", F.col("totalamount").cast("double")
        )
    elif "price" in lower_cols and "quantity" in lower_cols:
        df_features = df_features.withColumn(
            "TotalAmount", (F.col("price") * F.col("quantity")).cast("double")
        )
    else:
        raise ValueError(
            "Missing TotalAmount; expected total_sales, totalamount, or price+quantity"
        )

    df_features = df_features.withColumn("Month", F.month("OrderDate")).withColumn(
        "DayOfWeek", F.dayofweek("OrderDate")
    )

    predictions = model.transform(df_features)
    print(f"✅ Generated predictions for {predictions.count()} records.")
except Exception as e:
    print(f"❌ ERROR: Failed to process data at {HDFS_DATA_PATH}: {e}")
    spark.stop()
    sys.exit(1)

# ─── 4. MONITORING: Performance Check ──────────────────────────────────────
print("[MONITOR] Evaluating model accuracy...")
evaluator = RegressionEvaluator(labelCol="TotalAmount", predictionCol="prediction")

rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

# Threshold logic: Flag if accuracy drops (RMSE increases)
# (Adjust 1000.0 based on your specific TotalAmount scales)
is_healthy = rmse < 1000.0 

# ─── 5. SAVE MONITORING MANIFEST ───────────────────────────────────────────
manifest = {
    "pipeline_run": datetime.now().isoformat(),
    "model_type": "Spark_ML_Random_Forest",
    "deployment_source": "Local_Disk_Volume",
    "metrics": {
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    },
    "monitoring": {
        "status": "HEALTHY" if is_healthy else "DEGRADED",
        "action_required": not is_healthy
    }
}

os.makedirs(os.path.dirname(MANIFEST_OUT), exist_ok=True)
with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=4)

print(f"\n✅ Phase 5 Complete. Manifest saved to: {MANIFEST_OUT}")
print(f"   Model Health: {'[OK]' if is_healthy else '[ALERT]'}")

spark.stop()