import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession


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
    spark = get_spark("DemandForecasting-Phase2-DataAnalyst")
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
    cleaned_path = f"{hdfs_uri}/user/data-engineer/demand_forecasting/cleaned_data"

    output_dir = os.getenv("OUTPUT_DIR", "/pipeline/data/analysis")
    os.makedirs(output_dir, exist_ok=True)

    df = spark.read.parquet(cleaned_path)
    print("Basic describe() stats:")
    describe_df = df.describe()
    describe_df.show()

    pd_df = df.toPandas()
    describe_pd = describe_df.toPandas()
    describe_pd.to_csv(f"{output_dir}/describe_stats.csv", index=False)

    daily_sales = pd_df.groupby("order_date")["total_sales"].sum().reset_index()
    daily_sales = daily_sales.sort_values("order_date")

    plt.figure(figsize=(12, 6))
    sns.lineplot(x="order_date", y="total_sales", data=daily_sales)
    plt.title("Daily Total Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daily_sales_trend.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    product_sales = (
        pd_df.groupby("product_id")["total_sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    sns.barplot(x="product_id", y="total_sales", data=product_sales, ax=axes[0])
    axes[0].set_title("Total Sales by Product")
    axes[0].set_xlabel("Product ID")
    axes[0].set_ylabel("Total Sales")
    axes[0].tick_params(axis="x", rotation=45)

    store_sales = (
        pd_df.groupby("store_id")["total_sales"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    sns.barplot(x="store_id", y="total_sales", data=store_sales, ax=axes[1])
    axes[1].set_title("Total Sales by Store")
    axes[1].set_xlabel("Store ID")
    axes[1].set_ylabel("Total Sales")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sales_by_product_store.png")
    plt.close()

    print(f"Saved plots to: {output_dir}")
    spark.stop()


if __name__ == "__main__":
    main()
