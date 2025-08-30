# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing Pipeline
# MAGIC This notebook handles data loading, cleaning, and feature engineering

# COMMAND ----------

import sys
sys.path.append('/Workspace/Repos/your-repo/fraud-detection-bundle/src')

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer

# COMMAND ----------

# Get parameters
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

# Initialize config
config = Config()
config.CATALOG_NAME = catalog_name
config.SCHEMA_NAME = schema_name

logger = setup_logging()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

data_loader = DataLoader(config)
raw_df = data_loader.load_data()

print(f"Loaded data shape: {raw_df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Data

# COMMAND ----------

data_cleaner = DataCleaner()

# Assess data quality
clean_df = data_cleaner.assess_data(raw_df)

# Clean and validate
clean_df = data_cleaner.clean_and_validate(clean_df)

print(f"Cleaned data shape: {clean_df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Engineer Features

# COMMAND ----------

feature_engineer = FeatureEngineer(config)

# Basic feature engineering
feature_df = feature_engineer.engineer_basic_features(clean_df)

# Create temporal splits
train_df, val_df, test_df = feature_engineer.create_temporal_splits(feature_df)

print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Processed Data

# COMMAND ----------

# Save to Delta tables for next stage
train_spark_df = data_loader.spark.createDataFrame(train_df.to_pandas())
val_spark_df = data_loader.spark.createDataFrame(val_df.to_pandas())
test_spark_df = data_loader.spark.createDataFrame(test_df.to_pandas())

train_table = f"{catalog_name}.{schema_name}.processed_train_data"
val_table = f"{catalog_name}.{schema_name}.processed_val_data"
test_table = f"{catalog_name}.{schema_name}.processed_test_data"

train_spark_df.write.mode("overwrite").saveAsTable(train_table)
val_spark_df.write.mode("overwrite").saveAsTable(val_table)
test_spark_df.write.mode("overwrite").saveAsTable(test_table)

logger.info("Data preprocessing completed successfully!")
