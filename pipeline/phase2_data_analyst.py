import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --- Configuration & Paths ---------------------------------------------------
BASE_PATH = "/pipeline/data"
CLEANED_DATA_PATH = f"{BASE_PATH}/cleaned_data/amazon_cleaned.csv"
OUTPUT_DIR = f"{BASE_PATH}/analysis/figures"
SUMMARY_FILE = f"{BASE_PATH}/analysis/business_summary.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Spark Session ------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("Phase2_Amazon_Analyst")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# --- 1. Load Data -------------------------------------------------------------
df_spark = spark.read.csv(CLEANED_DATA_PATH, header=True, inferSchema=True)
print(f"[INFO] Analyzing {df_spark.count()} orders.")

# Convert to Pandas for full EDA
df = df_spark.toPandas()

# --- 2. Key Metrics Summary ---------------------------------------------------
summary_stats = df_spark.select("TotalAmount", "Quantity", "Discount", "ShippingCost").describe()
summary_stats.toPandas().to_csv(SUMMARY_FILE, index=False)

# --- 3. Category Performance --------------------------------------------------
category_analysis = (
    df_spark.groupBy("Category")
    .agg(
        F.sum("TotalAmount").alias("Total_Revenue"),
        F.count("OrderID").alias("Order_Count"),
        F.avg("TotalAmount").alias("Avg_Order_Value")
    )
    .orderBy(F.desc("Total_Revenue"))
)
category_analysis.show()

# --- 4. Monthly Trend Analysis ------------------------------------------------
monthly_trend = (
    df_spark.withColumn("YearMonth", F.date_format("OrderDate", "yyyy-MM"))
    .groupBy("YearMonth")
    .agg(F.sum("TotalAmount").alias("Revenue"))
    .orderBy("YearMonth")
)

# --- 5. Quarterly Aggregation -------------------------------------------------
# Full quarterly timeline (e.g. "2022 Q1", "2022 Q2" ...)
quarterly_trend = (
    df_spark.withColumn("Quarter", F.concat(
        F.year("OrderDate").cast("string"),
        F.lit(" Q"),
        F.quarter("OrderDate").cast("string")
    ))
    .groupBy("Quarter")
    .agg(F.sum("TotalAmount").alias("Revenue"))
    .orderBy("Quarter")
)
quarterly_pdf = quarterly_trend.toPandas()
quarterly_pdf.to_csv(f"{BASE_PATH}/analysis/quarterly_revenue.csv", index=False)
print("[INFO] Saved: quarterly_revenue.csv")

# Collapsed Q1-Q4 summary across all years
quarter_summary = (
    df_spark.withColumn("Quarter", F.concat(F.lit("Q"), F.quarter("OrderDate").cast("string")))
    .groupBy("Quarter")
    .agg(F.sum("TotalAmount").alias("Revenue"))
    .orderBy("Quarter")
)
quarter_summary_pdf = quarter_summary.toPandas()

# --- 6. Column Classification -------------------------------------------------
sns.set_theme(style="whitegrid")

num_cols = df.select_dtypes(include=["number"]).columns.to_list()
cat_cols = df.select_dtypes(include=["object"]).drop(
    columns=["ProductName"], errors="ignore"
).columns.to_list()

# --- 7. Outlier Statistics -> CSV ---------------------------------------------
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
print(f"[INFO] Outlier rows detected: {outlier_mask.sum()}")

outlier_stats = pd.DataFrame({
    "Q1": Q1,
    "Q3": Q3,
    "IQR": IQR,
    "Lower_Bound": Q1 - 1.5 * IQR,
    "Upper_Bound": Q3 + 1.5 * IQR,
    "Outlier_Count": ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).sum()
})
outlier_stats.to_csv(f"{BASE_PATH}/analysis/outlier_statistics.csv")
print("[INFO] Saved: outlier_statistics.csv")

# --- 8. Skewness & Kurtosis -> CSV --------------------------------------------
skew = df[num_cols].skew()
kurt = df[num_cols].kurtosis()
skew_kurt_df = pd.DataFrame({"Skewness": skew, "Kurtosis": kurt})
skew_kurt_df.to_csv(f"{BASE_PATH}/analysis/skewness_kurtosis.csv")
print("[INFO] Saved: skewness_kurtosis.csv")
print(skew_kurt_df)

# --- 9. Correlation Matrix -> CSV ---------------------------------------------
corr_matrix = df[num_cols].corr()
corr_matrix.to_csv(f"{BASE_PATH}/analysis/correlation_matrix.csv")
print("[INFO] Saved: correlation_matrix.csv")

# --- 10. Geographic Insights --------------------------------------------------
top_cities = (
    df_spark.groupBy("City")
    .agg(F.sum("TotalAmount").alias("City_Revenue"))
    .orderBy(F.desc("City_Revenue"))
    .limit(10)
)
top_cities.toPandas().to_csv(f"{BASE_PATH}/analysis/top_10_cities.csv", index=False)
print("[INFO] Saved: top_10_cities.csv")

# =============================================================================
# CHARTS (10 total)
# =============================================================================

# --- Chart 01: Revenue by Category -------------------------------------------
cat_pdf = category_analysis.toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(data=cat_pdf, x="Total_Revenue", y="Category",
            hue="Category", palette="viridis", legend=False)
plt.title("Total Revenue by Product Category")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_revenue_by_category.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 01_revenue_by_category.png")

# --- Chart 02: Sales Distribution --------------------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(df["TotalAmount"], bins=30, kde=True, color="blue")
plt.title("Distribution of Order Total Amounts")
plt.xlabel("Total Amount ($)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_sales_distribution.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 02_sales_distribution.png")

# --- Chart 03: Payment Method Pie --------------------------------------------
plt.figure(figsize=(8, 8))
df["PaymentMethod"].value_counts().plot.pie(
    autopct="%1.1f%%", colors=sns.color_palette("pastel")
)
plt.title("Payment Method Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_payment_methods.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 03_payment_methods.png")

# --- Chart 04: Histograms for All Numerical Columns --------------------------
columns = 3
rows = (len(num_cols) + columns - 1) // columns
fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
axes = axes.flatten()
plt.suptitle("Histograms - Numerical Columns", fontsize=16, y=1.01)
for i, col in enumerate(num_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
for ax in axes[len(num_cols):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_histograms_numerical.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 04_histograms_numerical.png")

# --- Chart 05: Countplots for Categorical Columns ----------------------------
columns = 3
rows = (len(cat_cols) + columns - 1) // columns
fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
axes = axes.flatten()
plt.suptitle("Countplots - Categorical Columns", fontsize=16, y=1.01)
for i, col in enumerate(cat_cols):
    sns.countplot(y=df[col], hue=df[col], ax=axes[i], palette="Set2", legend=False)
    axes[i].set_title(f"Distribution of {col}")
for ax in axes[len(cat_cols):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_countplots_categorical.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 05_countplots_categorical.png")

# --- Chart 06: Violin Plot - TotalAmount -------------------------------------
plt.figure(figsize=(10, 5))
sns.violinplot(y=df["TotalAmount"], color="skyblue")
plt.title("Spread of Total Amount")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_violin_total_amount.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 06_violin_total_amount.png")

# --- Chart 07: Correlation Matrix Heatmap ------------------------------------
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, linewidth=0.9, cmap="YlOrRd")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_correlation_matrix.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 07_correlation_matrix.png")

# --- Chart 08: Scatter Plots vs TotalAmount ----------------------------------
x_cols = [col for col in num_cols if col != "TotalAmount"]
y = "TotalAmount"
columns = 3
rows = (len(x_cols) + columns - 1) // columns
fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
axes = axes.flatten()
sample_df = df.sample(min(4500, len(df)), random_state=42)
for i, x in enumerate(x_cols):
    sns.scatterplot(data=sample_df, x=x, y=y, ax=axes[i], alpha=0.5)
    axes[i].set_title(f"{x} vs {y}")
for ax in axes[len(x_cols):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_scatter_vs_total_amount.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 08_scatter_vs_total_amount.png")

# --- Chart 09: Revenue by Quarter Bar Chart (Q1-Q4) --------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=quarter_summary_pdf, x="Quarter", y="Revenue",
            hue="Quarter", palette="coolwarm", legend=False)
plt.title("Total Revenue by Quarter (Q1-Q4)")
plt.xlabel("Quarter")
plt.ylabel("Total Revenue ($)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_revenue_by_quarter_bar.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 09_revenue_by_quarter_bar.png")

# --- Chart 10: Quarterly Revenue Trend Over Time Line Chart ------------------
plt.figure(figsize=(14, 5))
plt.plot(quarterly_pdf["Quarter"], quarterly_pdf["Revenue"],
         marker="o", color="steelblue", linewidth=2)
plt.xticks(rotation=45, ha="right")
plt.title("Quarterly Revenue Trend Over Time")
plt.xlabel("Quarter")
plt.ylabel("Total Revenue ($)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_quarterly_revenue_trend.png", bbox_inches="tight")
plt.close()
print("[INFO] Saved: 10_quarterly_revenue_trend.png")

# --- Done --------------------------------------------------------------------
print(f"\n[DONE] All 10 charts saved in:     {OUTPUT_DIR}")
print(f"[DONE] Business summary:           {SUMMARY_FILE}")
print(f"[DONE] Skewness/Kurtosis:          {BASE_PATH}/analysis/skewness_kurtosis.csv")
print(f"[DONE] Outlier statistics:         {BASE_PATH}/analysis/outlier_statistics.csv")
print(f"[DONE] Correlation matrix:         {BASE_PATH}/analysis/correlation_matrix.csv")
print(f"[DONE] Quarterly revenue:          {BASE_PATH}/analysis/quarterly_revenue.csv")
print(f"[DONE] Top 10 cities:              {BASE_PATH}/analysis/top_10_cities.csv")

spark.stop()
