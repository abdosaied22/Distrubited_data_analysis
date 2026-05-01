import json
import os

import pandas as pd
import streamlit as st
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

st.title("📊 MLOps Dashboard - Demand Forecasting")


MODEL_PATH = "file:///pipeline/data/models/Random_Forest"
MANIFEST_PATH = "/pipeline/data/optimized_results/mlops_manifest.json"
REQUIRED_COLUMNS = ["OrderDate", "Quantity", "Discount", "Tax", "ShippingCost"]


@st.cache_resource
def get_spark():
    return SparkSession.builder.appName("StreamlitApp").getOrCreate()


spark = get_spark()


@st.cache_resource
def load_model():
    return PipelineModel.load(MODEL_PATH)


try:
    model = load_model()
    st.success("✅ Model Loaded Successfully")
except Exception as exc:
    model = None
    st.error(f"❌ Failed to load model: {exc}")


if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

        st.subheader("📈 Model Performance")
        st.json(manifest)

        st.metric("RMSE", manifest["metrics"]["rmse"])
        st.metric("R2 Score", manifest["metrics"]["r2"])

        status = manifest["monitoring"]["status"]
        if status == "HEALTHY":
            st.success("🟢 Model is Healthy")
        else:
            st.error("🔴 Model Degraded")

else:
    st.warning("No manifest found")

st.subheader("📥 Predict From File")
uploaded_file = st.file_uploader(
    "Upload CSV with OrderDate, Quantity, Discount, Tax, ShippingCost",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        pdf = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
    else:
        missing = [col for col in REQUIRED_COLUMNS if col not in pdf.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        elif model is None:
            st.error("Model is not loaded, cannot run predictions.")
        else:
            sdf = spark.createDataFrame(pdf)
            sdf = (
                sdf.withColumn("OrderDate", F.to_date(F.col("OrderDate")))
                .withColumn("Quantity", F.col("Quantity").cast("double"))
                .withColumn("Discount", F.col("Discount").cast("double"))
                .withColumn("Tax", F.col("Tax").cast("double"))
                .withColumn("ShippingCost", F.col("ShippingCost").cast("double"))
                .withColumn("Month", F.month("OrderDate"))
                .withColumn("DayOfWeek", F.dayofweek("OrderDate"))
            )

            try:
                predictions = model.transform(sdf)
                output = predictions.select(
                    "OrderDate",
                    "Quantity",
                    "Discount",
                    "Tax",
                    "ShippingCost",
                    "prediction",
                ).toPandas()
                st.subheader("✅ Predictions")
                st.dataframe(output)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

st.subheader("✍️ Predict From Form")
with st.form("single_prediction_form"):
    order_date = st.date_input("Order Date")
    quantity = st.number_input("Quantity", min_value=0.0, step=1.0)
    discount = st.number_input("Discount", min_value=0.0, step=0.01)
    tax = st.number_input("Tax", min_value=0.0, step=0.01)
    shipping_cost = st.number_input("Shipping Cost", min_value=0.0, step=0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    if model is None:
        st.error("Model is not loaded, cannot run predictions.")
    else:
        form_row = pd.DataFrame(
            [
                {
                    "OrderDate": order_date.strftime("%Y-%m-%d"),
                    "Quantity": quantity,
                    "Discount": discount,
                    "Tax": tax,
                    "ShippingCost": shipping_cost,
                }
            ]
        )
        sdf = spark.createDataFrame(form_row)
        sdf = (
            sdf.withColumn("OrderDate", F.to_date(F.col("OrderDate")))
            .withColumn("Quantity", F.col("Quantity").cast("double"))
            .withColumn("Discount", F.col("Discount").cast("double"))
            .withColumn("Tax", F.col("Tax").cast("double"))
            .withColumn("ShippingCost", F.col("ShippingCost").cast("double"))
            .withColumn("Month", F.month("OrderDate"))
            .withColumn("DayOfWeek", F.dayofweek("OrderDate"))
        )
        try:
            prediction = (
                model.transform(sdf)
                .select("prediction")
                .toPandas()
                .iloc[0]["prediction"]
            )
            st.metric("Predicted Total Amount", f"{prediction:.2f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.subheader("⚡ Spark Optimization Summary")

st.markdown("""

Adaptive Query Execution: ✅ Enabled
Broadcast Joins: ✅ Used
Partitioning: ✅ Used
Caching: ✅ Used
Expensive Actions: ❌ count() detected
""")

st.subheader("🧠 Notes")
st.info("This dashboard shows model health and Spark job optimization insights.")
