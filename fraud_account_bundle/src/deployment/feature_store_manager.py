import polars as pl
import pyarrow as pa
from databricks.feature_store import FeatureStoreClient
from pyspark.sql import SparkSession
from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class FeatureStoreManager:
    def __init__(self, config: Config):
        self.config = config
        self.fs = FeatureStoreClient()
        self.spark = SparkSession.getActiveSession()
    
    def polars_to_spark(self, pl_df: pl.DataFrame):
        """Convert Polars DataFrame to Spark DataFrame"""
        arrow_table = pl_df.to_arrow()
        spark_df = self.spark.createDataFrame(pa.Table.to_pandas(arrow_table))
        return spark_df
    
    def save_features_to_store(self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame):
        """Save feature sets to Databricks Feature Store"""
        logger.info("Saving features to Feature Store...")
        
        # Convert DataFrames
        train_spark_df = self.polars_to_spark(train_df)
        val_spark_df = self.polars_to_spark(val_df)
        test_spark_df = self.polars_to_spark(test_df)
        
        # Define table names
        train_table_name = f"{self.config.CATALOG_NAME}.{self.config.SCHEMA_NAME}.train_features"
        val_table_name = f"{self.config.CATALOG_NAME}.{self.config.SCHEMA_NAME}.val_features"
        test_table_name = f"{self.config.CATALOG_NAME}.{self.config.SCHEMA_NAME}.test_features"
        
        # Create tables
        try:
            self.fs.create_table(
                name=train_table_name,
                primary_keys=["index"],
                df=train_spark_df,
                schema=train_spark_df.schema,
                description="Training features for fraud detection"
            )
            logger.info(f"Created training feature table: {train_table_name}")
            
            self.fs.create_table(
                name=val_table_name,
                primary_keys=["index"],
                df=val_spark_df,
                schema=val_spark_df.schema,
                description="Validation features for fraud detection"
            )
            logger.info(f"Created validation feature table: {val_table_name}")
            
            self.fs.create_table(
                name=test_table_name,
                primary_keys=["index"],
                df=test_spark_df,
                schema=test_spark_df.schema,
                description="Test features for fraud detection"
            )
            logger.info(f"Created test feature table: {test_table_name}")
            
        except Exception as e:
            logger.warning(f"Some feature tables may already exist: {e}")
            logger.info("Continuing with existing tables...")
