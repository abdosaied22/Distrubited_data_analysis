import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from minio import Minio
from pyspark.sql import SparkSession


def run_phase(script_name: str, retries: int = 1, retry_delay_sec: int = 20) -> None:
    for attempt in range(1, retries + 1):
        try:
            print(f"\n=== Running {script_name} (attempt {attempt}/{retries}) ===")
            subprocess.run([sys.executable, script_name], check=True)
            print(f"=== Completed {script_name} ===")
            return
        except subprocess.CalledProcessError:
            if attempt >= retries:
                raise
            print(
                f"{script_name} failed on attempt {attempt}. "
                f"Retrying in {retry_delay_sec}s to allow services to become ready..."
            )
            time.sleep(retry_delay_sec)


def get_spark(app_name: str) -> SparkSession:
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
    spark_master = os.getenv("SPARK_MASTER", "local[*]")

    return (
        SparkSession.builder.appName(app_name)
        .master(spark_master)
        .config("spark.submit.deployMode", "client")
        .config("spark.hadoop.fs.defaultFS", hdfs_uri)
        .config(
            "spark.hadoop.dfs.replication", os.getenv("HDFS_CONF_dfs_replication", "1")
        )
        .getOrCreate()
    )


def build_minio_client() -> Minio:
    endpoint_raw = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    parsed = urlparse(endpoint_raw)

    if parsed.scheme:
        endpoint = parsed.netloc
        secure = parsed.scheme == "https"
    else:
        endpoint = endpoint_raw
        secure = False

    return Minio(
        endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "password2026"),
        secure=secure,
    )


def upload_cleaned_data_to_minio() -> None:
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
    hdfs_cleaned_path = f"{hdfs_uri}/user/data-engineer/demand_forecasting/cleaned_data"
    minio_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")
    minio_prefix = os.getenv("MINIO_SILVER_PREFIX", "demand_forecasting/cleaned_data")

    spark = get_spark("DemandForecasting-HDFS-To-MinIO-Silver")
    temp_dir = Path(tempfile.mkdtemp(prefix="cleaned_data_"))

    try:
        print(f"Reading from HDFS: {hdfs_cleaned_path}")
        df = spark.read.parquet(hdfs_cleaned_path)
        row_count = df.count()
        print(f"Rows read from HDFS: {row_count}")

        local_parquet_dir = temp_dir / "cleaned_data"
        local_parquet_uri = f"file://{local_parquet_dir}"
        print(f"Writing temporary parquet locally: {local_parquet_uri}")
        df.write.mode("overwrite").parquet(local_parquet_uri)

        client = build_minio_client()
        if not client.bucket_exists(minio_bucket):
            client.make_bucket(minio_bucket)

        uploaded_files = 0
        for file_path in local_parquet_dir.rglob("*"):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(local_parquet_dir).as_posix()
            object_name = f"{minio_prefix}/{relative_path}"
            client.fput_object(minio_bucket, object_name, str(file_path))
            uploaded_files += 1

        print(
            f"Uploaded {uploaded_files} files to MinIO bucket '{minio_bucket}' "
            f"under prefix '{minio_prefix}'"
        )
    finally:
        spark.stop()
        shutil.rmtree(temp_dir, ignore_errors=True)


def upload_hdfs_path_to_minio(
    *,
    hdfs_path: str,
    minio_bucket: str,
    minio_prefix: str,
    app_name: str,
) -> None:
    spark = get_spark(app_name)
    temp_dir = Path(tempfile.mkdtemp(prefix="hdfs_export_"))

    try:
        print(f"Reading from HDFS: {hdfs_path}")
        df = spark.read.parquet(hdfs_path)
        rows = df.count()
        print(f"Rows read from HDFS: {rows}")

        local_parquet_dir = temp_dir / "export"
        local_parquet_uri = f"file://{local_parquet_dir}"
        print(f"Writing temporary parquet locally: {local_parquet_uri}")
        df.write.mode("overwrite").parquet(local_parquet_uri)

        client = build_minio_client()
        if not client.bucket_exists(minio_bucket):
            client.make_bucket(minio_bucket)

        uploaded_files = 0
        for file_path in local_parquet_dir.rglob("*"):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(local_parquet_dir).as_posix()
            object_name = f"{minio_prefix}/{relative_path}"
            client.fput_object(minio_bucket, object_name, str(file_path))
            uploaded_files += 1

        print(
            f"Uploaded {uploaded_files} files to MinIO bucket '{minio_bucket}' "
            f"under prefix '{minio_prefix}'"
        )
    finally:
        spark.stop()
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    # HDFS can be reachable before DataNode is fully registered. Retry phase 1 writes.
    run_phase("phase1_data_engineer.py", retries=6, retry_delay_sec=20)
    upload_cleaned_data_to_minio()
    run_phase("phase2_data_analyst.py")
    run_phase("phase3_ml_engineer.py")
    upload_hdfs_path_to_minio(
        hdfs_path=f"{os.getenv('CORE_CONF_fs_defaultFS', 'hdfs://namenode:9000')}/user/data-engineer/demand_forecasting/models/linear_regression_model/data",
        minio_bucket=os.getenv("MINIO_GOLD_BUCKET", "gold"),
        minio_prefix=os.getenv(
            "MINIO_GOLD_MODEL_PREFIX",
            "demand_forecasting/models/linear_regression_model",
        ),
        app_name="DemandForecasting-HDFS-Model-To-MinIO-Gold",
    )
    run_phase("phase4_big_data_engineer.py")
    upload_hdfs_path_to_minio(
        hdfs_path=f"{os.getenv('CORE_CONF_fs_defaultFS', 'hdfs://namenode:9000')}/user/data-engineer/demand_forecasting/optimized_data",
        minio_bucket=os.getenv("MINIO_GOLD_BUCKET", "gold"),
        minio_prefix=os.getenv(
            "MINIO_GOLD_OPT_PREFIX", "demand_forecasting/optimized_data"
        ),
        app_name="DemandForecasting-HDFS-Optimized-To-MinIO-Gold",
    )
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
