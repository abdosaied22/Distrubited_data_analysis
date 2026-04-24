import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)


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


def main() -> None:
    spark = get_spark("DemandForecasting-Phase1-DataEngineer")
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
    local_data_dir = os.getenv("LOCAL_DATA_DIR", "/pipeline/data")

    print(f"Spark master: {spark.sparkContext.master}")
    print(f"HDFS URI: {hdfs_uri}")

    num_rows = 10000

    products = [f"product_{i}" for i in range(10)]
    stores = [f"store_{i}" for i in range(5)]

    rows = []
    for _ in range(num_rows):
        day_offset = int(np.random.randint(0, 365))
        order_date = np.datetime64("2022-01-01") + np.timedelta64(day_offset, "D")
        quantity = float(np.random.randint(1, 100))
        price = float(np.round(np.random.uniform(5.0, 500.0), 2))
        rows.append(
            (
                str(order_date),
                str(np.random.choice(products)),
                str(np.random.choice(stores)),
                quantity,
                price,
                int(np.random.choice([0, 1], p=[0.8, 0.2])),
            )
        )

    # Inject missing values to preserve the original cleaning logic.
    for idx in np.random.choice(num_rows, 50, replace=False):
        r = list(rows[int(idx)])
        r[3] = None
        rows[int(idx)] = tuple(r)
    for idx in np.random.choice(num_rows, 20, replace=False):
        r = list(rows[int(idx)])
        r[4] = None
        rows[int(idx)] = tuple(r)

    schema = StructType(
        [
            StructField("order_date", StringType(), True),
            StructField("product_id", StringType(), True),
            StructField("store_id", StringType(), True),
            StructField("quantity", DoubleType(), True),
            StructField("price", DoubleType(), True),
            StructField("is_promo", IntegerType(), True),
        ]
    )

    spark_df = spark.createDataFrame(rows, schema=schema)
    spark_df = spark_df.withColumn("order_date", to_date(col("order_date")))

    mean_quantity = spark_df.agg({"quantity": "mean"}).collect()[0][0]
    median_price = spark_df.approxQuantile("price", [0.5], 0.01)[0]
    spark_df_cleaned = spark_df.fillna(
        {"quantity": mean_quantity, "price": median_price}
    )
    spark_df_cleaned = spark_df_cleaned.withColumn(
        "total_sales", col("quantity") * col("price")
    )

    smoke_path = f"{hdfs_uri}/tmp/demand_forecasting_smoke_test"
    cleaned_path = f"{hdfs_uri}/user/data-engineer/demand_forecasting/cleaned_data"
    local_cleaned_path = f"file://{local_data_dir}/cleaned_data"

    os.makedirs(local_data_dir, exist_ok=True)

    spark.range(1).write.mode("overwrite").parquet(smoke_path)
    spark_df_cleaned.write.mode("overwrite").parquet(cleaned_path)
    spark_df_cleaned.write.mode("overwrite").parquet(local_cleaned_path)

    print(f"Smoke test path: {smoke_path}")
    print(f"Cleaned data path: {cleaned_path}")
    print(f"Local cleaned data path: {local_cleaned_path}")
    print(f"Rows written: {spark_df_cleaned.count()}")

    spark.stop()


if __name__ == "__main__":
    main()
