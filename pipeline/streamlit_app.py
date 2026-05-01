import streamlit as st
import json
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

st.title("📊 MLOps Dashboard - Demand Forecasting")



MODEL_PATH = "/pipeline/data/models/Random_Forest"
MANIFEST_PATH = "/pipeline/data/optimized_results/mlops_manifest.json"


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
except:
    st.error("❌ Failed to load model")


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

st.subheader("⚡ Spark Optimization Summary")

st.markdown("""

Adaptive Query Execution: ✅ Enabled
Broadcast Joins: ✅ Used
Partitioning: ⚠️ Needs tuning
Caching: ❌ Not used properly
Expensive Actions: ❌ count() detected
""")

st.subheader("🧠 Notes")
st.info("This dashboard shows model health and Spark job optimization insights.")