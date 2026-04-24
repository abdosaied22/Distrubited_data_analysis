import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    dayofmonth,
    dayofweek,
    month,
    year,
    weekofyear,
    lag,
)
from pyspark.sql.window import Window
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


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
    spark = get_spark("DemandForecasting-Phase3-MLEngineer")
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")

    cleaned_path = f"{hdfs_uri}/user/data-engineer/demand_forecasting/cleaned_data"
    model_root = f"{hdfs_uri}/user/data-engineer/demand_forecasting/models"
    featurizer_path = f"{model_root}/featurizer_pipeline_model"
    lr_model_path = f"{model_root}/linear_regression_model"

    local_model_root = os.getenv("ML_LOCAL_MODEL_DIR", "/pipeline/data/ml/models")
    local_featurizer_path = f"file://{local_model_root}/featurizer_pipeline_model"
    local_lr_model_path = f"file://{local_model_root}/linear_regression_model"

    output_dir = os.getenv("ML_OUTPUT_DIR", "/pipeline/data/ml")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(local_model_root, exist_ok=True)

    df = spark.read.parquet(cleaned_path)

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

    indexer_product = StringIndexer(inputCol="product_id", outputCol="product_idx")
    indexer_store = StringIndexer(inputCol="store_id", outputCol="store_idx")
    encoder_product = OneHotEncoder(inputCol="product_idx", outputCol="product_vec")
    encoder_store = OneHotEncoder(inputCol="store_idx", outputCol="store_vec")

    feature_cols = [
        "day_of_month",
        "day_of_week",
        "month",
        "year",
        "week_of_year",
        "lag_1_day_qty",
        "product_vec",
        "store_vec",
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    featurizer_pipeline = Pipeline(
        stages=[
            indexer_product,
            indexer_store,
            encoder_product,
            encoder_store,
            assembler,
        ]
    )

    featurizer_model = featurizer_pipeline.fit(daily_agg_df)
    ml_df = featurizer_model.transform(daily_agg_df)
    model_data = ml_df.select("features", "daily_quantity")

    train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(
        featuresCol="features",
        labelCol="daily_quantity",
        maxIter=10,
        regParam=0.3,
        elasticNetParam=0.8,
    )
    lr_model = lr.fit(train_data)

    predictions = lr_model.transform(test_data)
    evaluator_rmse = RegressionEvaluator(
        labelCol="daily_quantity", predictionCol="prediction", metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="daily_quantity", predictionCol="prediction", metricName="r2"
    )

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    featurizer_model.write().overwrite().save(featurizer_path)
    lr_model.write().overwrite().save(lr_model_path)
    featurizer_model.write().overwrite().save(local_featurizer_path)
    lr_model.write().overwrite().save(local_lr_model_path)

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "train_rows": train_data.count(),
        "test_rows": test_data.count(),
        "featurizer_path": featurizer_path,
        "lr_model_path": lr_model_path,
        "local_featurizer_path": local_featurizer_path,
        "local_lr_model_path": local_lr_model_path,
    }

    with open(f"{output_dir}/ml_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Model training complete")
    print(json.dumps(metrics, indent=2))

    spark.stop()


if __name__ == "__main__":
    main()
