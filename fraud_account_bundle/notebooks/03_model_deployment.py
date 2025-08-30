# Databricks notebook source
# MAGIC %md
# MAGIC # Model Deployment Pipeline
# MAGIC This notebook handles model registration and feature store integration

# COMMAND ----------

import sys
sys.path.append('/Workspace/Repos/your-repo/fraud-detection-bundle/src')
import polars as pl
import mlflow

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.deployment.model_deployer import ModelDeployer
from src.deployment.feature_store_manager import FeatureStoreManager

# COMMAND ----------

# Get parameters
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
model_name = dbutils.widgets.get("model_name")

# Initialize config
config = Config()
config.CATALOG_NAME = catalog_name
config.SCHEMA_NAME = schema_name
config.MODEL_NAME = model_name

logger = setup_logging()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data and Model Artifacts

# COMMAND ----------

# Load processed data
train_table = f"{catalog_name}.{schema_name}.processed_train_data"
val_table = f"{catalog_name}.{schema_name}.processed_val_data"
test_table = f"{catalog_name}.{schema_name}.processed_test_data"

train_spark_df = spark.read.table(train_table)
val_spark_df = spark.read.table(val_table)
test_spark_df = spark.read.table(test_table)

train_df = pl.from_pandas(train_spark_df.toPandas())
val_df = pl.from_pandas(val_spark_df.toPandas())
test_df = pl.from_pandas(test_spark_df.toPandas())

# Get training results from previous task
try:
    best_params = dbutils.jobs.taskValues.get(taskKey="model_training", key="best_params")
    model_metrics = dbutils.jobs.taskValues.get(taskKey="model_training", key="model_metrics")
    cat_features = dbutils.jobs.taskValues.get(taskKey="model_training", key="cat_features")
except:
    logger.warning("Could not retrieve task values, using defaults")
    best_params = {}
    model_metrics = {}
    cat_features = []

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Model

# COMMAND ----------

# Load the best model from MLflow
model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
model = mlflow.catboost.load_model(model_uri)

# Deploy model
deployer = ModelDeployer(config)
X_test_pd = test_df.drop("fraud_type").to_pandas()
y_test = test_df.select("fraud_type").to_series().to_numpy()

model_info = deployer.deploy_model(
    model=model,
    X_test=X_test_pd,
    y_test=y_test,
    cat_features=cat_features,
    best_params=best_params,
    model_stage="Staging",
    register_model=True
)

print(f"Model deployed: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Store Integration

# COMMAND ----------

feature_store_manager = FeatureStoreManager(config)
feature_store_manager.save_features_to_store(train_df, val_df, test_df)

# COMMAND ----------

logger.info("Model deployment completed successfully!")
