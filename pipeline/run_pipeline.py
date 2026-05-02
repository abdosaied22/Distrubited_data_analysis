import os
import subprocess
import sys
import time

from helper import (
    build_minio_client,
    upload_hdfs_path_to_minio,
    upload_local_csv_to_minio,
)


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


def main() -> None:
    # HDFS can be reachable before DataNode is fully registered. Retry phase 1 writes.
    run_phase("phase1_data_engineer.py", retries=6, retry_delay_sec=20)
    upload_local_csv_to_minio(
        local_csv_path="/pipeline/data/cleaned_data/amazon_cleaned.csv",
        minio_bucket=os.getenv("MINIO_SILVER_BUCKET", "silver"),
        minio_prefix=os.getenv(
            "MINIO_SILVER_PREFIX", "demand_forecasting/cleaned_data"
        ),
        app_name="DemandForecasting-LocalCSV-To-MinIO-Silver",
    )
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
    run_phase("phase5_mlops_engineer.py")

    # Upload Monitoring Manifest to MinIO Gold
    client = build_minio_client()
    manifest_local = "/pipeline/data/logs/pipeline_manifest.json"
    if os.path.exists(manifest_local):
        client.fput_object("gold", "monitoring/pipeline_manifest.json", manifest_local)
        print("[INFO] Uploaded MLOps Manifest to MinIO Gold.")

    print("\n✅ E2E Pipeline completed successfully.")

    streamlit_port = os.getenv("STREAMLIT_PORT", "8501")
    print(f"[INFO] Starting Streamlit on port {streamlit_port}...")
    subprocess.run(
        [
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.address=0.0.0.0",
            f"--server.port={streamlit_port}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
