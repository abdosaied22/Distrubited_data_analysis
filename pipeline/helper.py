import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from minio import Minio
from pyspark.sql import SparkSession


def get_spark(app_name: str) -> SparkSession:
    hdfs_uri = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
    spark_master = os.getenv("SPARK_MASTER", "local[*]")

    return (
        SparkSession.builder.appName(app_name)
        .master(spark_master)
        .config("spark.submit.deployMode", "client")
        .config("spark.hadoop.fs.defaultFS", hdfs_uri)
        .config(
            "spark.hadoop.dfs.replication",
            os.getenv("HDFS_CONF_dfs_replication", "1"),
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


def upload_local_csv_to_minio(
    *,
    local_csv_path: str,
    minio_bucket: str,
    minio_prefix: str,
    app_name: str,
) -> None:
    spark = get_spark(app_name)
    temp_dir = Path(tempfile.mkdtemp(prefix="local_csv_export_"))

    try:
        local_uri = f"file://{Path(local_csv_path).resolve()}"
        print(f"Reading local CSV: {local_uri}")
        df = spark.read.csv(local_uri, header=True, inferSchema=True)
        rows = df.count()
        print(f"Rows read from local CSV: {rows}")

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


def upload_local_path_to_hdfs(
    *,
    local_path: str,
    hdfs_path: str,
    data_format: str,
    app_name: str,
) -> None:
    spark = get_spark(app_name)

    try:
        local_uri = f"file://{Path(local_path).resolve()}"
        print(f"Reading local data: {local_uri} ({data_format})")

        if data_format == "csv":
            df = spark.read.csv(local_uri, header=True, inferSchema=True)
        elif data_format == "parquet":
            df = spark.read.parquet(local_uri)
        else:
            raise ValueError("data_format must be 'csv' or 'parquet'")

        rows = df.count()
        print(f"Rows read from local data: {rows}")
        print(f"Writing to HDFS: {hdfs_path}")
        df.write.mode("overwrite").parquet(hdfs_path)
    finally:
        spark.stop()
