# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training Pipeline
# MAGIC This notebook handles model training and hyperparameter optimization

# COMMAND ----------

import sys
sys.path.append('/Workspace/Repos/your-repo/fraud-detection-bundle/src')
import polars as pl
import mlflow

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.training.model_trainer import ModelTrainer
from src.training.hyperparameter_optimizer import HyperparameterOptimizer
from src.training.model_evaluator import ModelEvaluator

# COMMAND ----------

# Get parameters
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
model_name = dbutils.widgets.get("model_name")
experiment_name = dbutils.widgets.get("experiment_name")

# Initialize config
config = Config()
config.CATALOG_NAME = catalog_name
config.SCHEMA_NAME = schema_name
config.MODEL_NAME = model_name
config.EXPERIMENT_NAME = experiment_name

logger = setup_logging()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Processed Data

# COMMAND ----------

train_table = f"{catalog_name}.{schema_name}.processed_train_data"
test_table = f"{catalog_name}.{schema_name}.processed_test_data"

train_spark_df = spark.read.table(train_table)
test_spark_df = spark.read.table(test_table)

train_df = pl.from_pandas(train_spark_df.toPandas())
test_df = pl.from_pandas(test_spark_df.toPandas())

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features

# COMMAND ----------

model_trainer = ModelTrainer(config)
X_train, y_train, X_test, y_test, cat_features = model_trainer.prepare_features_for_training(train_df, test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MLflow

# COMMAND ----------

# Set MLflow experiment
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Optimization

# COMMAND ----------

optimizer = HyperparameterOptimizer(config)
optimization_results = optimizer.optimize(X_train, y_train, X_test, y_test, cat_features)

best_params = optimization_results["best_params"]
final_model = optimization_results["final_model"]

print(f"Best parameters: {best_params}")
print(f"Best CV AUC: {optimization_results['best_cv_auc']:.4f}")
print(f"Test AUC: {optimization_results['test_auc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(final_model, X_test, y_test)

print("Final Model Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Training Results

# COMMAND ----------

# Store results for deployment stage
dbutils.jobs.taskValues.set(key="best_params", value=best_params)
dbutils.jobs.taskValues.set(key="model_metrics", value=metrics)
dbutils.jobs.taskValues.set(key="cat_features", value=cat_features)

logger.info("Model training completed successfully!")
