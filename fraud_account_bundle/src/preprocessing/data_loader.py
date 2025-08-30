import polars as pl
from pyspark.sql import SparkSession
from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.spark = SparkSession.getActiveSession()
        
    def load_data(self) -> pl.DataFrame:
        """Load data from Delta table and convert to Polars"""
        try:
            logger.info(f"Reading Delta table: {self.config.full_table_name}")
            
            # Read Delta table
            spark_df = self.spark.read.table(self.config.full_table_name)
            
            # Convert to Pandas then Polars for memory efficiency
            pandas_df = spark_df.toPandas()
            polars_df = pl.from_pandas(pandas_df)
            
            logger.info(f"Data loaded successfully. Shape: {polars_df.shape}")
            return polars_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
