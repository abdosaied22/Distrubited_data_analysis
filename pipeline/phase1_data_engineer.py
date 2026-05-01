import os
from pathlib import Path

from pyspark.sql import functions as F

from helper import get_spark

# ─── Environment & Paths ────────────────────────────────────
BASE_PATH = "/pipeline/data"
RAW_PATH = f"{BASE_PATH}/Amazon.csv"
LOCAL_OUTPUT_PATH = f"{BASE_PATH}/cleaned_data/amazon_cleaned.csv"

HDFS_URI = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
HDFS_OUTPUT = f"{HDFS_URI}/user/data-engineer/amazon_project/cleaned_data"

os.makedirs(os.path.dirname(LOCAL_OUTPUT_PATH), exist_ok=True)

# ─── Spark Session ──────────────────────────────────────────
spark = get_spark("Amazon_Clean_Only_Phase1")

# ─── 1. Load Data ───────────────────────────────────────────
raw_uri = f"file://{Path(RAW_PATH).resolve()}"
df = spark.read.csv(raw_uri, header=True, inferSchema=True)

# ─── 2. Cleaning & Column Pruning ───────────────────────────
cols_to_drop = ["CustomerID", "ProductID"]

df_clean = (
    df.drop(*cols_to_drop)
    .dropDuplicates(["OrderID"])
    .dropna(subset=["OrderID", "OrderDate", "TotalAmount"])
    .withColumn("OrderDate", F.to_date("OrderDate", "yyyy-MM-dd"))
    # Numeric Sanity
    .withColumn("TotalAmount", F.abs(F.col("TotalAmount")))
    .withColumn(
        "Quantity", F.when(F.col("Quantity") < 1, 1).otherwise(F.col("Quantity"))
    )
)

df_clean = df_clean.fillna({"Discount": 0.0, "Tax": 0.0, "ShippingCost": 0.0})
# ─── 2.1 Outlier Removal (IQR Method) ───────────────────────
quantiles = df_clean.approxQuantile("TotalAmount", [0.25, 0.75], 0.05)
Q1 = quantiles[0]
Q3 = quantiles[1]
IQR = Q3 - Q1

# Define Bounds (Standard 1.5 * IQR rule)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(
    f"[INFO] Removing outliers in TotalAmount outside: ({lower_bound:.2f}, {upper_bound:.2f})"
)

# Filter the DataFrame
df_clean = df_clean.filter(
    (F.col("TotalAmount") >= lower_bound) & (F.col("TotalAmount") <= upper_bound)
)
# ─── 3. Storage Layer Execution ─────────────────────────────

# A. HDFS
print(f"Writing to HDFS: {HDFS_OUTPUT}")
df_clean.write.mode("overwrite").parquet(HDFS_OUTPUT)

# B. Local Filesystem (CSV)
print(f"Writing to Local CSV: {LOCAL_OUTPUT_PATH}")

df.toPandas().to_csv(LOCAL_OUTPUT_PATH, index=False)


# ─── Done ───────────────────────────────────────────────────
print("✅ Cleaning complete. Data synced to HDFS and Local CSV.")
spark.stop()
