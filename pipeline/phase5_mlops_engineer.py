import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofmonth, dayofweek, month, year, weekofyear, lag
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel


def get_spark(app_name: str) -> SparkSession:
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
    yarn_rm = os.getenv("YARN_CONF_yarn_resourcemanager_hostname", "resourcemanager")
    spark_master = os.getenv("SPARK_MASTER", "yarn")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(spark_master)
        .config("spark.submit.deployMode", "client")
        .config("spark.hadoop.fs.defaultFS", hdfs_uri)
        .config("spark.hadoop.yarn.resourcemanager.hostname", yarn_rm)
        .config("spark.hadoop.dfs.replication", os.getenv("HDFS_CONF_dfs_replication", "1"))
        .getOrCreate()
    )
    return spark


def main() -> None:
    spark = get_spark("DemandForecasting-Phase5-MLOps")
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")

    cleaned_path = f"{hdfs_uri}/user/data-engineer/demand_forecasting/cleaned_data"
    model_root = f"{hdfs_uri}/user/data-engineer/demand_forecasting/models"
    featurizer_path = f"{model_root}/featurizer_pipeline_model"
    lr_model_path = f"{model_root}/linear_regression_model"

    output_dir = os.getenv("OUTPUT_DIR", "output")
    os.makedirs(output_dir, exist_ok=True)

    df = spark.read.parquet(cleaned_path)

    data_quality = {
        "row_count": df.count(),
        "null_quantity": df.filter(col("quantity").isNull()).count(),
        "null_price": df.filter(col("price").isNull()).count(),
        "null_total_sales": df.filter(col("total_sales").isNull()).count(),
    }

    featurizer_model = PipelineModel.load(featurizer_path)
    lr_model = LinearRegressionModel.load(lr_model_path)

    daily_agg_df = (
        df.groupBy("order_date", "product_id", "store_id")
        .agg({"quantity": "sum", "total_sales": "sum"})
        .withColumnRenamed("sum(quantity)", "daily_quantity")
        .withColumnRenamed("sum(total_sales)", "daily_total_sales")
    )

    window_spec = Window.partitionBy("product_id", "store_id").orderBy("order_date")
    daily_agg_df = (
        daily_agg_df.withColumn("day_of_month", dayofmonth(col("order_date")))
        .withColumn("day_of_week", dayofweek(col("order_date")))
        .withColumn("month", month(col("order_date")))
        .withColumn("year", year(col("order_date")))
        .withColumn("week_of_year", weekofyear(col("order_date")))
        .withColumn("lag_1_day_qty", lag(col("daily_quantity"), 1).over(window_spec))
        .fillna(0, subset=["lag_1_day_qty"])
    )

    transformed = featurizer_model.transform(daily_agg_df)
    predictions = lr_model.transform(transformed)

    prediction_preview = predictions.select(
        "order_date", "product_id", "store_id", "daily_quantity", "prediction"
    )

    preview_rows = [row.asDict() for row in prediction_preview.limit(10).collect()]
    report = {
        "data_quality": data_quality,
        "prediction_preview_count": len(preview_rows),
    }

    with open(f"{output_dir}/mlops_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("MLOps checks complete")
    print(json.dumps(report, indent=2, default=str))
    prediction_preview.show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
