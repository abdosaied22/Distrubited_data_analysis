import json
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --- Paths & Configuration (CRITICAL FIX FOR DOCKER VOLUMES) ---
# Most Docker setups map the local folder to /pipeline/data
# We ensure RESULTS_DIR is inside the mapped volume path
SHARED_VOLUME_PATH = "/pipeline/data" 
RESULTS_DIR = os.path.join(SHARED_VOLUME_PATH, "optimized_results")
CLEANED_INPUT = os.path.join(SHARED_VOLUME_PATH, "cleaned_data/amazon_cleaned.csv")
MASTER_METRICS_JSON = os.path.join(RESULTS_DIR, "optimization_metrics.json")

# Create the directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[INFO] Created directory: {RESULTS_DIR}")

spark = (
    SparkSession.builder
    .appName("Phase4_BigDataEngineer_Amazon")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.hadoop.dfs.replication", "3")
    .getOrCreate()
)

master_metrics = {
    "timings": [],
    "partitions": {},
    "replication": {
        "factor": spark.conf.get("spark.hadoop.dfs.replication"),
        "strategy": "HDFS Fault Tolerance"
    },
    "skew_report": []
}

def timed_log(section, scenario, operation, fn):
    t0 = time.time()
    result = fn()
    duration = round(time.time() - t0, 4)
    master_metrics["timings"].append({
        "section": section,
        "scenario": scenario,
        "operation": operation,
        "seconds": duration
    })
    return result

# --- 1. Load & Partition Summary ---
print(f"[INFO] Loading data from: {CLEANED_INPUT}")
df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(CLEANED_INPUT)

def capture_partitions(df, label):
    parts = df.rdd.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()
    master_metrics["partitions"][label] = {
        "count": len(parts),
        "sample_distribution": parts[:5]
    }

capture_partitions(df_raw, "before_repartition")

# --- 2. Repartition (Distributed Processing) ---
df_optimized = df_raw.repartition(8, "State")
capture_partitions(df_optimized, "after_repartition")

# --- 3. Caching ---
timed_log("Caching", "before", "uncached_count", lambda: df_raw.count())
df_optimized.cache()
timed_log("Caching", "after", "first_cached_count", lambda: df_optimized.count())
timed_log("Caching", "after", "second_cached_count", lambda: df_optimized.count())

# --- 4. Skew Detection ---
skew_data = df_raw.groupBy("Category").count().orderBy(F.desc("count")).limit(5).collect()
master_metrics["skew_report"] = [{"category": r["Category"], "count": r["count"]} for r in skew_data]

# --- 5. Broadcast Join ---
state_meta = df_raw.select("State").distinct().withColumn("Region", F.lit("Standard"))
timed_log("Join", "before", "shuffle_join", lambda: df_raw.join(state_meta, "State").count())
timed_log("Join", "after", "broadcast_join", lambda: df_raw.join(F.broadcast(state_meta), "State").count())

# --- 6. Final Aggregation & Output ---
output_csv_path = os.path.join(RESULTS_DIR, "final_optimized_data_csv")
optimized_agg = df_optimized.groupBy("State", "Category").agg(F.sum("TotalAmount").alias("Revenue"))

# We use coalesce(1) so it creates a single CSV file inside the folder
timed_log("Write", "after", "optimized_csv_write", 
          lambda: optimized_agg.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_csv_path))

# --- 7. Save Consolidated Metrics (The JSON Fix) ---
try:
    with open(MASTER_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=4)
    print(f"\n✅ SUCCESS: All metrics saved to local device at: {MASTER_METRICS_JSON}")
except Exception as e:
    print(f"❌ ERROR: Failed to save JSON: {e}")

spark.stop()