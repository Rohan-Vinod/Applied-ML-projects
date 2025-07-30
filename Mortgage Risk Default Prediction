import pyspark
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, to_date, datediff, expr, lit, date_sub, round, when, count as spark_count, min as spark_min, avg as spark_avg, stddev as spark_stddev, countDistinct, sum as spark_sum, max as spark_max, year, last_day, add_months, sequence, explode
from pyspark.sql.functions import months_between, when, month, mean, stddev, min as spark_min, max as spark_max, last
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number


# Start Spark session
spark = SparkSession.builder.appName("TechnicalTask").getOrCreate()
# Makes names for directories of data-sets to be analysed
base_path = "C:/Users/vinod/OneDrive/Desktop/Technical Task"

#Loads and reads dataset
customers_path = f"{base_path}/Customers.csv"
defaults_path = f"{base_path}/Defaults.csv"
mortgages_path = f"{base_path}/Mortgages.csv"
lookup_path = f"{base_path}/MasterLookup.csv"

def read_csv(path):
    return (
        spark.read
             .option("header", "true")
             .option("inferSchema", "true")
             .option("nullValue", "null")   # treat both as NULL
             .option("nanValue", "NA")
             .csv(path)
    )

customers_df = read_csv(customers_path)
defaults_df  = read_csv(defaults_path)
mortgages_df = read_csv(mortgages_path)
lookup_df    = read_csv(lookup_path)

#Establishes NA back into property regions after being classed as NULL
mortgages_df = mortgages_df.withColumn(
    "PropertyRegion",
    when(col("PropertyRegion").isNull(), lit("NA")).otherwise(col("PropertyRegion"))
)

#Establishes t-1 data types.
customers_df = (
    customers_df
        .withColumn("ObservationDate", to_date(col("ObservationDate")))
)

defaults_df = (
    defaults_df
        .withColumn("DefaultDate", to_date(col("DefaultDate")))
        .withColumn("ReportDate",  to_date(col("ReportDate")))
)

mortgages_df = (
    mortgages_df
        .withColumn("LoanOriginationDate", to_date(col("LoanOriginationDate")))
        .withColumn("MaturityDate",        to_date(col("MaturityDate")))
)

for name, df in [
        ("customers", customers_df),
        ("defaults",  defaults_df),
        ("mortgages", mortgages_df),
        ("lookup",    lookup_df),
]:
    print(f"\n*** {name.upper()} ***")
    df.printSchema()
    print(f"Rows: {df.count():,}")
    df.show(5, truncate=False)

from pyspark.sql.functions import last_day

customers_df = customers_df.withColumn(
    "SnapshotMonth", last_day(col("ObservationDate"))
)

defaults_labeled = (
    defaults_df
        .filter(col("DefaultDate").isNotNull())  # only real defaults
        .withColumn("DefaultMonth", last_day(col("DefaultDate")))
        .select("DebtID", "DefaultMonth")
        .withColumn("target", lit(1))  # label = 1 for defaults
)

#Takes latest snapshot from customer observations
window_spec = Window.partitionBy("ConsumerID", "SnapshotMonth").orderBy(col("ObservationDate").desc())
customers_deduped = (
    customers_df
    .withColumn("rownum", row_number().over(window_spec))
    .filter(col("rownum") == 1)
    .drop("rownum")
)

#Joins in debt ID to consumer ID
cust_loans = customers_deduped.join(lookup_df, on="ConsumerID", how="left")

#Joins mortgage info onto customer info
cust_loans = cust_loans.join(mortgages_df, on="DebtID", how="left")

#Creates window that we want to predict for
feature_df = (
    cust_loans.join(
        defaults_labeled,
        (cust_loans["DebtID"] == defaults_labeled["DebtID"]) &
        (last_day(add_months(cust_loans["SnapshotMonth"], 1)) == defaults_labeled["DefaultMonth"]),
        how="left"
    )
    .drop(defaults_labeled["DebtID"])
    .drop("DefaultMonth")
    .fillna({"target": 0})
)

#Finds first default date for each loan
first_defaults = (
    defaults_df
    .filter(col("DefaultDate").isNotNull())
    .groupBy("DebtID")
    .agg(spark_min("DefaultDate").alias("FirstDefaultDate"))
)

#Joins first default date into feature
feature_df = feature_df.join(first_defaults, on="DebtID", how="left")

#Drops any records were snapshot occurs AFTER loan has happened
feature_df = feature_df.filter(
    (col("SnapshotMonth") < col("FirstDefaultDate")) | col("FirstDefaultDate").isNull()
)

#Also drops foreclosure value, as not relevant to default rate.
feature_df.printSchema()
feature_df = feature_df.drop("ForeclosureValue")


#-----FEATURE ENGINEERING-----#

feature_df = (
    feature_df
    # Static loan features
    .withColumn("LoanToValue", round(col("OriginalBalance") / col("OriginalPropertyValue"), 4))
    .withColumn("LoanAgeMonths", round(months_between(col("SnapshotMonth"), col("LoanOriginationDate"))))
    .withColumn("LoanTermMonths", round(months_between(col("MaturityDate"), col("LoanOriginationDate"))))
    .withColumn("TimeToMaturityMonths", round(months_between(col("MaturityDate"), col("SnapshotMonth"))))
    .withColumn("SnapshotMonthNum", month(col("SnapshotMonth")))
    .withColumn(
        "LoanProgressRatio",
        when(col("LoanTermMonths") > 0,
            round(col("LoanAgeMonths") / col("LoanTermMonths"), 4)
        ).otherwise(None)
    )
    .withColumn(
        "LTVBand",
        when(col("LoanToValue") <= 0.5, "<=50%")
        .when(col("LoanToValue") <= 0.75, "51–75%")
        .when(col("LoanToValue") <= 0.9, "76–90%")
        .when(col("LoanToValue") <= 1.0, "91–100%")
        .otherwise(">100%")
    )
    .withColumn(
        "LoanAgeYears",
        round(col("LoanAgeMonths") / 12, 2)
    )

    # Loan-customer interaction features (with divide-by-zero guard)
    .withColumn("DebtToIncome", when(col("IndexedTotalIncome") > 0,
                                     round(col("OriginalBalance") / col("IndexedTotalIncome"), 4)))
    .withColumn("ExposureToIncome", when(col("IndexedTotalIncome") > 0,
                                         round(col("TotalExposure") / col("IndexedTotalIncome"), 4)))
)

#From prior model, LTV values with very high balances
ltv_errors = feature_df.filter(col("LoanToValue") > 2.5).count()
print("Number of rows with LoanToValue > 2.5:", ltv_errors)
max_ltv = feature_df.agg(spark_max("LoanToValue")).collect()[0][0]
print("Maximum LoanToValue:", max_ltv)

#After confirming high check counts of over 11119LTV as a max and only 2980 with a high LTV, classes as invalid
#Therefore, removed from features set
feature_df = feature_df.filter(col("LoanToValue") <= 2.5)
#Quick check to confirm minimal impact and sane values
print("New row count after LTV filter:", feature_df.count())
feature_df.agg(spark_max("LoanToValue")).show()

#General sanity check 1
#for c in ["LoanToValue", "LoanAgeMonths", "LoanTermMonths", "TimeToMaturityMonths", "DebtToIncome", "ExposureToIncome"]:
#    feature_df.select(spark_count(when(col(c).isNull(), 1)).alias(f"{c}_nulls")).show()
#    print("Total rows in feature_df:", feature_df.count())

#General sanity check 2
#defaulted_loans = defaults_df.filter(col("DefaultDate").isNotNull()).select("DebtID").distinct().count()
#print("Number of distinct loans that have defaulted:", defaulted_loans)

# List of defaulted loans
defaulted_loans_df = defaults_df.filter(col("DefaultDate").isNotNull()).select("DebtID", "DefaultDate")

# Add t-1 month
defaulted_loans_df = defaulted_loans_df.withColumn("SnapshotMonth", last_day(add_months(col("DefaultDate"), -1)))

# Join to customer snapshots
joined = defaulted_loans_df.join(feature_df.select("DebtID", "SnapshotMonth"),
                                 on=["DebtID", "SnapshotMonth"], how="left")

##Sanity check: Count how many match
#with_snapshot = joined.filter(col("SnapshotMonth").isNotNull()).count()
#
#print("Defaults with usable t-1 snapshot:", with_snapshot)
#print("Defaults missing t-1 snapshot:", 1447 - with_snapshot)

#Aggregates data to avoid dominance of non-defaulted loans
aggregated_df = (
    feature_df
    .groupBy("DebtID")
    .agg(
        spark_max("target").alias("target"),  # if it defaulted in any snapshot, mark it
        spark_avg("LoanToValue").alias("AvgLoanToValue"),
        spark_avg("LoanAgeMonths").alias("AvgLoanAgeMonths"),
        spark_avg("LoanTermMonths").alias("AvgLoanTermMonths"),
        spark_avg("TimeToMaturityMonths").alias("AvgTimeToMaturity"),
        spark_avg("DebtToIncome").alias("AvgDebtToIncome"),
        spark_avg("ExposureToIncome").alias("AvgExposureToIncome"),
        spark_avg("IndexedTotalIncome").alias("AvgIncome"),
        stddev("IndexedTotalIncome").alias("IncomeVolatility"),
        spark_avg("InterestRate").alias("AvgInterestRate"),
        spark_avg("ConsumerAge").alias("AvgConsumerAge"),
        spark_avg("SnapshotMonthNum").alias("AvgSnapshotMonth")
    )
)

#Sanity check for count of rows after aggregation
print("New row count after aggregation:", aggregated_df.count())

##-----PROPERTY REGION ANALYSIS-----#
#Done in this section to take advantage of one loan per row aggregated data
#Finds region lookup
region_lookup = (
    feature_df
    .select("DebtID", "PropertyRegion")
    .dropna(subset=["PropertyRegion"])
    .dropDuplicates(["DebtID"])  # assumes one region per loan
)

#Joins back
feature_df_updated = aggregated_df.join(region_lookup, on="DebtID", how="left")

#Outputs risk table for property regions
print("\n--- PropertyRegion Default Rates (Loan-level after aggregation) ---\n")
(
    feature_df_updated.groupBy("PropertyRegion")
    .agg(
        spark_count("*").alias("LoanCount"),
        spark_avg("target").alias("DefaultRate")
    )
    .orderBy("DefaultRate", ascending=False)
    .show(100, truncate=False)
)

#Drops after aggregation
feature_df_updated = feature_df_updated.drop("DebtID")
feature_df_updated = feature_df_updated.drop("ConsumerID")

#Good practice to cache after dropping columns
feature_df_updated = feature_df_updated.cache()

#Sanitycheck to confirm columns
#print(feature_df.columns)
#

#Establishes columns for correlation
correlation_cols = [
    "AvgDebtToIncome",
    "AvgExposureToIncome",
    "AvgLoanToValue",
    "AvgLoanAgeMonths",
    "AvgLoanTermMonths",
    "AvgTimeToMaturity",
    "AvgIncome",
    "IncomeVolatility",
    "AvgInterestRate",
    "AvgConsumerAge",
    "AvgSnapshotMonth",
    "target"
]

#Converts correlation dataset to pandas for correlation matrix
correlation_df = feature_df_updated.select(correlation_cols).toPandas()


corr_matrix = correlation_df.corr()
# Print correlation of each feature with the target, sorted
print(corr_matrix["target"].sort_values(ascending=False))

##Key features to implement:
# -IncomeVolatility
# -AvgTimeToMaturity
# -AvgLoanTermMonths
# -AvgConsumerAge
# -AvgInterestRate
# -AvgLoanAgeMonths
# -AvgIncome

##Can analyse in rate tables:
# -DebtToIncome
# -ExposureToIncome
# -AvgLoanToValue

##-----RISK TABLES-----##

#--BANDING--#

# DebtToIncomeBand
feature_df_updated = feature_df_updated.withColumn(
    "DebtToIncomeBand",
    when(col("AvgDebtToIncome") <= 0.5, "≤0.5x")
    .when(col("AvgDebtToIncome") <= 1.0, "0.5–1x")
    .when(col("AvgDebtToIncome") <= 1.5, "1–1.5x")
    .when(col("AvgDebtToIncome") <= 2.0, "1.5–2x")
    .when(col("AvgDebtToIncome") <= 2.5, "2–2.5x")
    .when(col("AvgDebtToIncome") <= 3.0, "2.5–3x")
    .when(col("AvgDebtToIncome") <= 3.5, "3–3.5x")
    .when(col("AvgDebtToIncome") <= 4.0, "3.5–4x")
    .when(col("AvgDebtToIncome") <= 4.5, "4–4.5x")
    .when(col("AvgDebtToIncome") <= 5.0, "4.5–5x")
    .when(col("AvgDebtToIncome") <= 5.5, "5–5.5x")
    .when(col("AvgDebtToIncome") <= 6.0, "5.5–6x")
    .when(col("AvgDebtToIncome") <= 7.0, "6–7x")
    .when(col("AvgDebtToIncome") <= 8.0, "7–8x")
    .when(col("AvgDebtToIncome") <= 10.0, "8–10x")
    .otherwise(">10x")
)

# ExposureToIncomeBand
feature_df_updated = feature_df_updated.withColumn(
    "ExposureToIncomeBand",
    when(col("AvgExposureToIncome") <= 2.0, "≤2x")
    .when(col("AvgExposureToIncome") <= 3.0, "2–3x")
    .when(col("AvgExposureToIncome") <= 4.0, "3–4x")
    .when(col("AvgExposureToIncome") <= 5.0, "4–5x")
    .when(col("AvgExposureToIncome") <= 6.0, "5–6x")
    .when(col("AvgExposureToIncome") <= 7.0, "6–7x")
    .when(col("AvgExposureToIncome") <= 8.0, "7–8x")
    .when(col("AvgExposureToIncome") <= 9.0, "8–9x")
    .when(col("AvgExposureToIncome") <= 10.0, "9–10x")
    .when(col("AvgExposureToIncome") <= 11.0, "10–11x")
    .when(col("AvgExposureToIncome") <= 12.0, "11–12x")
    .otherwise(">12x")
)

# LoanToValueBand
feature_df_updated = feature_df_updated.withColumn(
    "LoanToValueBand",
    when(col("AvgLoanToValue") <= 0.55, "≤55%")
    .when(col("AvgLoanToValue") <= 0.60, "56–60%")
    .when(col("AvgLoanToValue") <= 0.65, "61–65%")
    .when(col("AvgLoanToValue") <= 0.70, "66–70%")
    .when(col("AvgLoanToValue") <= 0.75, "71–75%")
    .when(col("AvgLoanToValue") <= 0.80, "76–80%")
    .when(col("AvgLoanToValue") <= 0.85, "81–85%")
    .when(col("AvgLoanToValue") <= 0.90, "86–90%")
    .when(col("AvgLoanToValue") <= 0.95, "91–95%")
    .when(col("AvgLoanToValue") <= 1.00, "96–100%")
    .when(col("AvgLoanToValue") <= 1.05, "101–105%")
    .when(col("AvgLoanToValue") <= 1.10, "106–110%")
    .when(col("AvgLoanToValue") <= 1.15, "111–115%")
    .when(col("AvgLoanToValue") <= 1.20, "116–120%")
    .when(col("AvgLoanToValue") <= 1.25, "121–125%")
    .when(col("AvgLoanToValue") <= 1.30, "126–130%")
    .otherwise(">130%")
)

#OUTPUT TO VISUALISE RISK TABLE#
# Loan-level banding summary for selected rate bands
rate_bands = ["DebtToIncomeBand", "ExposureToIncomeBand", "LoanToValueBand"]

for band in rate_bands:
    print(f"\nLoan-level summary for {band}:\n")
    (
        feature_df_updated.groupBy(band)
        .agg(
            spark_count("*").alias("LoanCount"),
            spark_avg("target").alias("DefaultRate")
        )
        .orderBy(band)
        .show(truncate=False)
    )

##After analysing, good baseline model seems like logistic regression, with potential for XGBoost/Gradient Boosting
##For more accurate model.

# -----------------------
# LOGISTIC REGRESSION FEATURE SETUP
# -----------------------

#Categorical (binned or segmented) features – use StringIndexer + OneHotEncoder
#These capture known nonlinear threshold effects shown in rate tables
categorical_features = [
    "DebtToIncomeBand",       # From AvgDebtToIncome
    "ExposureToIncomeBand",   # From AvgExposureToIncome
    "LoanToValueBand",         # From AvgLoanToValue
    "PropertyRegion"          # From PropertyRegion
]

#Continuous (raw numeric) features – kept as is
#These generally show linear or monotonic relationship with target
numerical_features = [
    "AvgLoanAgeMonths",
    "AvgLoanTermMonths",
    "AvgTimeToMaturity",
    "AvgIncome",
    "AvgInterestRate",
    "AvgConsumerAge",
    "IncomeVolatility"
]

#Cleaning up NaN values
feature_df_updated = feature_df_updated.dropna(subset=numerical_features)


##Step 1 - HotEncoder for categorical features

indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_Index") for col in categorical_features]
encoders = [OneHotEncoder(inputCol=f"{col}_Index", outputCol=f"{col}_Vec") for col in categorical_features]

##Step 2 - Makes vector assembly of both numerical and categorical features (from encoder)
feature_assembler = VectorAssembler(
    inputCols=[f"{col}_Vec" for col in categorical_features] + numerical_features,
    outputCol="features"
)

##-----LOGISTIC REGRESSION MODEL SET-UP-----##


lr = LogisticRegression(labelCol="target", featuresCol="features")
pipeline = Pipeline(stages=indexers + encoders + [feature_assembler, lr])
train_data, test_data = feature_df_updated.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train_data)
predictions = model.transform(test_data)

##---EVALUATION OF LOGISTIC REGRESSION MODEL---##

#AUC Evaluation

evaluator = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Test AUC: {auc:.4f}")

#CONFUSION MATRIX Evaluation

# Add a prediction label using 0.5 threshold
# Convert the vector 'probability' into an array, then extract class-1 probability
predictions = predictions.withColumn("probability_1", vector_to_array(col("probability"))[1])
predictions = predictions.withColumn("prediction_label", when(col("probability_1") >= 0.5, 1).otherwise(0))

confusion_counts = (
    predictions
    .groupBy("target", "prediction_label")
    .count()
    .toPandas()
    .pivot(index="target", columns="prediction_label", values="count")
    .fillna(0)
    .astype(int)
)

#Determines True and False Positives and Negatives
TP = confusion_counts.loc[1, 1] if 1 in confusion_counts.columns and 1 in confusion_counts.index else 0
TN = confusion_counts.loc[0, 0] if 0 in confusion_counts.columns and 0 in confusion_counts.index else 0
FP = confusion_counts.loc[0, 1] if 1 in confusion_counts.columns and 0 in confusion_counts.index else 0
FN = confusion_counts.loc[1, 0] if 0 in confusion_counts.columns and 1 in confusion_counts.index else 0

# Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nConfusion Matrix:\n{confusion_counts}")
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1_score:.4f}")




##-----GRADIENT BOOSTING TREES-----##
gbt = GBTClassifier(labelCol="target", featuresCol="features", maxIter=50, maxDepth=5, seed=42)

#Sets up pipeline and train/test parameters, like logistic regression
pipeline_gbt = Pipeline(stages=indexers + encoders + [feature_assembler, gbt])
train_data, test_data = feature_df_updated.randomSplit([0.8, 0.2], seed=42)

model_gbt = pipeline_gbt.fit(train_data)
predictions_gbt = model_gbt.transform(test_data)

##AUC##

evaluator = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_gbt = evaluator.evaluate(predictions_gbt)
print(f"GBT AUC: {auc_gbt:.4f}")

##Correlation matrix, to compare with logistic regression
predictions_gbt = predictions_gbt.withColumn("probability_1", vector_to_array(col("probability"))[1])
predictions_gbt = predictions_gbt.withColumn("prediction_label", when(col("probability_1") >= 0.5, 1).otherwise(0))

confusion_counts = (
    predictions_gbt
    .groupBy("target", "prediction_label")
    .count()
    .toPandas()
    .pivot(index="target", columns="prediction_label", values="count")
    .fillna(0)
    .astype(int)
)

#Determines True and False Positives and Negatives
TP = confusion_counts.loc[1, 1] if 1 in confusion_counts.columns and 1 in confusion_counts.index else 0
TN = confusion_counts.loc[0, 0] if 0 in confusion_counts.columns and 0 in confusion_counts.index else 0
FP = confusion_counts.loc[0, 1] if 1 in confusion_counts.columns and 0 in confusion_counts.index else 0
FN = confusion_counts.loc[1, 0] if 0 in confusion_counts.columns and 1 in confusion_counts.index else 0

# Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("GBT CONFUSION MATRIX:")
print(f"\nConfusion Matrix:\n{confusion_counts}")
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1_score:.4f}")
