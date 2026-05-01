import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─── Configuration & Paths ───────────────────────────────────────────────────
BASE_PATH = "/pipeline/data"
# Reading the cleaned CSV we just created in Phase 1
CLEANED_DATA_PATH = f"{BASE_PATH}/cleaned_data/amazon_cleaned.csv"
OUTPUT_DIR = f"{BASE_PATH}/analysis/figures"
SUMMARY_FILE = f"{BASE_PATH}/analysis/business_summary.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Spark Session ────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder.appName("Phase2_Amazon_Analyst")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
# We use the local CSV for EDA and visualization
cleaned_uri = f"file://{Path(CLEANED_DATA_PATH).resolve()}"
df = spark.read.csv(cleaned_uri, header=True, inferSchema=True)
print(f"[INFO] Analyzing {df.count()} orders.")

# ─── 2. Key Metrics Summary ───────────────────────────────────────────────────
summary_stats = df.select(
    "TotalAmount", "Quantity", "Discount", "ShippingCost"
).describe()
summary_stats.toPandas().to_csv(SUMMARY_FILE, index=False)

# ─── 3. Category Performance (EDA) ───────────────────────────────────────────
category_analysis = (
    df.groupBy("Category")
    .agg(
        F.sum("TotalAmount").alias("Total_Revenue"),
        F.count("OrderID").alias("Order_Count"),
        F.avg("TotalAmount").alias("Avg_Order_Value"),
    )
    .orderBy(F.desc("Total_Revenue"))
)
category_analysis.show()

# ─── 4. Monthly Trend Analysis ──────────────────────────────────────────────
# Creating a monthly trend to see seasonality
monthly_trend = (
    df.withColumn("YearMonth", F.date_format("OrderDate", "yyyy-MM"))
    .groupBy("YearMonth")
    .agg(F.sum("TotalAmount").alias("Revenue"))
    .orderBy("YearMonth")
)

# Analyze trends and patterns in sales data
monthly_trend_pdf = monthly_trend.toPandas()
monthly_trend_pdf["Revenue_Change_Pct"] = (
    monthly_trend_pdf["Revenue"].pct_change().fillna(0) * 100
)
monthly_trend_pdf.to_csv(f"{BASE_PATH}/analysis/monthly_trend.csv", index=False)

# ─── 5. Visualizations (Pandas/Seaborn) ──────────────────────────────────────
# Switch to Pandas for Plotting
pdf = df.toPandas()
sns.set_theme(style="whitegrid")

# Figure 1: Revenue by Category
plt.figure(figsize=(12, 6))
cat_pdf = category_analysis.toPandas()
sns.barplot(data=cat_pdf, x="Total_Revenue", y="Category", palette="viridis")
plt.title("Total Revenue by Product Category")
plt.savefig(f"{OUTPUT_DIR}/01_revenue_by_category.png")
plt.close()

# Figure 4: Monthly Revenue Trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_trend_pdf, x="YearMonth", y="Revenue", marker="o")
plt.title("Monthly Revenue Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_monthly_revenue_trend.png")
plt.close()

# Figure 2: Sales Distribution
plt.figure(figsize=(10, 6))
sns.histplot(pdf["TotalAmount"], bins=30, kde=True, color="blue")
plt.title("Distribution of Order Total Amounts")
plt.xlabel("Total Amount ($)")
plt.savefig(f"{OUTPUT_DIR}/02_sales_distribution.png")
plt.close()

# Figure 3: Payment Method Preference
plt.figure(figsize=(8, 8))
pdf["PaymentMethod"].value_counts().plot.pie(
    autopct="%1.1f%%", colors=sns.color_palette("pastel")
)
plt.title("Payment Method Distribution")
plt.ylabel("")
plt.savefig(f"{OUTPUT_DIR}/03_payment_methods.png")
plt.close()

# ─── 6. Geographic Insights ──────────────────────────────────────────────────
top_cities = (
    df.groupBy("City")
    .agg(F.sum("TotalAmount").alias("City_Revenue"))
    .orderBy(F.desc("City_Revenue"))
    .limit(10)
)
top_cities.toPandas().to_csv(f"{BASE_PATH}/analysis/top_10_cities.csv", index=False)

# ─── Done ─────────────────────────────────────────────────────────────────────
print(f"✅ EDA Complete. Figures saved in: {OUTPUT_DIR}")
print(f"✅ Business summary saved in: {SUMMARY_FILE}")

spark.stop()
