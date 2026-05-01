import os
import sys
import joblib
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ─── Environment Configuration ───────────────────────────────────────────────
HDFS_URI = os.getenv("CORE_CONF_fs_defaultFS", "hdfs://namenode:9000")
# Using the same HDFS path as your run_pipeline.py
INPUT_PATH = f"{HDFS_URI}/user/data-engineer/amazon_project/cleaned_data"
LOCAL_ANALYSIS_PATH = "/pipeline/data/analysis"
MODEL_SAVE_BASE = "/pipeline/data/models"

os.makedirs(LOCAL_ANALYSIS_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_BASE, exist_ok=True)

spark = SparkSession.builder.appName("Phase3_ML_Production").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ─── 1. Load & Prep ──────────────────────────────────────────────────────────
try:
    print(f"[INFO] Reading from HDFS: {INPUT_PATH}")
    df = spark.read.parquet(INPUT_PATH)
except Exception as e:
    print(f"CRITICAL ERROR: Path not found in HDFS. Ensure Phase 1 ran correctly: {e}")
    sys.exit(1)

# Feature Engineering: Adding Month and DayOfWeek
df_ml = df.withColumn("Month", F.month("OrderDate")) \
          .withColumn("DayOfWeek", F.dayofweek("OrderDate")) \
          .fillna(0)

# ─── 2. Setup Evaluation ─────────────────────────────────────────────────────
# We will track RMSE, MSE, and R2
evaluator_rmse = RegressionEvaluator(labelCol="TotalAmount", predictionCol="prediction", metricName="rmse")
evaluator_mse = RegressionEvaluator(labelCol="TotalAmount", predictionCol="prediction", metricName="mse")
evaluator_r2 = RegressionEvaluator(labelCol="TotalAmount", predictionCol="prediction", metricName="r2")

FEATURE_COLS = ["Quantity", "Discount", "Tax", "ShippingCost", "Month", "DayOfWeek"]
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)
accuracy_metrics = []

# ─── 3. Train & Evaluate Models ──────────────────────────────────────────────
models = {
    "Linear_Regression": LinearRegression(featuresCol="scaledFeatures", labelCol="TotalAmount"),
    "Random_Forest": RandomForestRegressor(featuresCol="scaledFeatures", labelCol="TotalAmount"),
    "Gradient_Boosting": GBTRegressor(featuresCol="scaledFeatures", labelCol="TotalAmount")
}

for name, model_obj in models.items():
    print(f"[INFO] Training {name}...")
    pipeline = Pipeline(stages=[assembler, scaler, model_obj])
    fit_model = pipeline.fit(train_df)
    
    # Save Model to local disk (which is your mapped volume)
    fit_model.write().overwrite().save(f"{MODEL_SAVE_BASE}/{name}")
    
    # Predictions & Metrics
    preds = fit_model.transform(test_df)
    rmse = evaluator_rmse.evaluate(preds)
    mse = evaluator_mse.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)
    
    accuracy_metrics.append({
        "Model": name,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2
    })

# ─── 4. ARIMA (Time Series) ──────────────────────────────────────────────────
print("[INFO] Processing ARIMA...")
from statsmodels.tsa.arima.model import ARIMA

ts_data = df_ml.groupBy("OrderDate").agg(F.sum("TotalAmount").alias("Sales")).toPandas()
ts_data['OrderDate'] = pd.to_datetime(ts_data['OrderDate'])
ts_data = ts_data.set_index('OrderDate').sort_index().asfreq('D').fillna(0)

if len(ts_data) > 10:
    train_size = int(len(ts_data) * 0.8)
    train_series = ts_data["Sales"].iloc[:train_size]
    test_series = ts_data["Sales"].iloc[train_size:]

    model_arima = ARIMA(train_series, order=(5, 1, 0))
    results_arima = model_arima.fit()
    joblib.dump(results_arima, f"{MODEL_SAVE_BASE}/arima_model.pkl")

    forecast = results_arima.forecast(steps=len(test_series))
    actual = test_series.to_numpy()
    predicted = forecast.to_numpy()
    errors = predicted - actual
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    variance = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 0.0 if variance == 0 else float(1 - (np.sum(errors ** 2) / variance))

    accuracy_metrics.append({"Model": "ARIMA", "RMSE": rmse, "MSE": mse, "R2": r2})
else:
    print("[WARN] Insufficient data for ARIMA")

# ─── 5. Save Accuracies to CSV ────────────────────────────────────────────────
accuracy_df = pd.DataFrame(accuracy_metrics)
accuracy_df.to_csv(f"{LOCAL_ANALYSIS_PATH}/model_accuracies.csv", index=False)

print("\n=== Model Performance Summary ===")
print(accuracy_df)
print(f"\n✅ Accuracies saved to {LOCAL_ANALYSIS_PATH}/model_accuracies.csv")

spark.stop()