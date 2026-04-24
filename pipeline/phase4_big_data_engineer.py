import os
import time
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def get_spark(app_name: str) -> SparkSession:
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
    yarn_rm = os.getenv("YARN_CONF_yarn_resourcemanager_hostname", "resourcemanager")
    spark_master = os.getenv("SPARK_MASTER", "yarn")

    spark = (
        SparkSession.builder.appName(app_name)
        .master(spark_master)
        .config("spark.submit.deployMode", "client")
        .config("spark.hadoop.fs.defaultFS", hdfs_uri)
        .config("spark.hadoop.yarn.resourcemanager.hostname", yarn_rm)
        .config(
            "spark.hadoop.dfs.replication", os.getenv("HDFS_CONF_dfs_replication", "1")
        )
        .getOrCreate()
    )
    return spark


def timed_count(df, label: str) -> None:
    start = time.perf_counter()
    rows = df.count()
    elapsed = time.perf_counter() - start
    print(f"{label}: rows={rows}, time={elapsed:.3f}s")


def timed_count_value(df) -> tuple[int, float]:
    start = time.perf_counter()
    rows = df.count()
    elapsed = time.perf_counter() - start
    return rows, elapsed


def main() -> None:
    spark = get_spark("DemandForecasting-Phase4-BigDataEngineer")
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
    cleaned_path = f"{hdfs_uri}/user/data-engineer/demand_forecasting/cleaned_data"
    optimized_path = f"{hdfs_uri}/user/data-engineer/demand_forecasting/optimized_data"

    local_data_dir = os.getenv("LOCAL_DATA_DIR", "/pipeline/data")
    local_bigdata_dir = os.getenv("BIGDATA_OUTPUT_DIR", "/pipeline/data/bigdata")
    local_optimized_path = f"file://{local_data_dir}/optimized_data"
    os.makedirs(local_bigdata_dir, exist_ok=True)

    df = spark.read.parquet(cleaned_path)

    print(f"Default partitions: {df.rdd.getNumPartitions()}")

    repartitioned_df = df.repartition(col("product_id"))
    print(f"Repartitioned partitions: {repartitioned_df.rdd.getNumPartitions()}")

    cached_df = repartitioned_df.cache()
    timed_count(cached_df, "First cached count")
    timed_count(cached_df, "Second cached count")
    row_count, cached_count_time = timed_count_value(cached_df)

    repartitioned_df.write.mode("overwrite").parquet(optimized_path)
    repartitioned_df.write.mode("overwrite").parquet(local_optimized_path)

    metrics = {
        "source_path": cleaned_path,
        "optimized_hdfs_path": optimized_path,
        "optimized_local_path": local_optimized_path,
        "rows": row_count,
        "default_partitions": df.rdd.getNumPartitions(),
        "optimized_partitions": repartitioned_df.rdd.getNumPartitions(),
        "cached_count_time_seconds": cached_count_time,
    }
    with open(
        os.path.join(local_bigdata_dir, "optimization_metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    cached_df.unpersist()
    print("Optimization phase completed")

    spark.stop()


if __name__ == "__main__":
    main()
