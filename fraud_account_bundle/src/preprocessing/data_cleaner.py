import polars as pl
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class DataCleaner:
    def __init__(self):
        self.label_encoders = {}
    
    def assess_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Assess data quality"""
        # Remove extra quotes from column names
        df = df.rename({col: col.strip('"') for col in df.columns})
        
        logger.info(f"Dataset Shape: {df.shape}")
        
        # Missing values
        missing_values = df.null_count()
        logger.info("Missing Values:")
        for col in df.columns:
            null_count = missing_values[col][0]
            if null_count > 0:
                logger.info(f"{col}: {null_count}")
        
        # Class distribution
        if "fraud_type" in df.columns:
            fraud_counts = df.select("fraud_type").to_series().value_counts()
            logger.info(f"Fraud Type Distribution:\n{fraud_counts}")
        
        return df
    
    def clean_and_validate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean and validate data with Polars"""
        logger.info("Starting data cleaning and validation...")
        
        initial_shape = df.shape
        logger.info(f"Initial dataset shape: {initial_shape}")
        
        # Convert date columns
        date_columns = ['statement_period', 'transaction_month']
        date_conversions = []
        for col in date_columns:
            if col in df.columns:
                date_conversions.append(
                    pl.col(col).str.strptime(pl.Date, "%Y-%m", strict=False).alias(col)
                )
        
        if date_conversions:
            df = df.with_columns(date_conversions)
        
        # Validate numeric columns
        numeric_cols = ['quantity', 'gross_revenue_in_usd', 'amount_due_in_usd']
        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        numeric_conversions = []
        for col in existing_numeric_cols:
            numeric_conversions.append(
                pl.col(col).cast(pl.Float64, strict=False).alias(col)
            )
        
        if numeric_conversions:
            df = df.with_columns(numeric_conversions)
        
        # Remove highly correlated features
        df = self._remove_correlated_features(df)
        
        # Remove columns with high missing values
        df = self._remove_sparse_columns(df, threshold=0.5)
        
        # Handle missing values
        if 'gross_revenue_in_usd' in df.columns:
            df = df.with_columns(
                pl.col('gross_revenue_in_usd').fill_null(0)
            )
        
        # Validate identifiers
        if 'isrc' in df.columns:
            df = df.with_columns(
                pl.col('isrc').str.to_uppercase().alias('isrc')
            )
        
        final_shape = df.shape
        logger.info(f"Final dataset shape: {final_shape}")
        
        return df
    
    def _remove_correlated_features(self, df: pl.DataFrame, threshold: float = 0.9) -> pl.DataFrame:
        """Remove highly correlated features"""
        logger.info("Identifying highly correlated features...")
        
        # Convert to pandas for correlation calculation
        corr_df = df.clone()
        categorical_cols = [col for col in corr_df.columns if corr_df[col].dtype == pl.Utf8]
        
        if categorical_cols:
            corr_pandas = corr_df.to_pandas()
            
            for col in categorical_cols:
                le = LabelEncoder()
                corr_pandas[col] = corr_pandas[col].fillna('missing').astype(str)
                encoded = le.fit_transform(corr_pandas[col])
                corr_pandas[col] = encoded
                self.label_encoders[col] = le
            
            corr_df = pl.from_pandas(corr_pandas)
        
        correlation_matrix = corr_df.to_pandas().corr()
        
        # Find highly correlated features
        to_drop = set()
        to_keep = set()
        for col1 in correlation_matrix.columns:
            for col2 in correlation_matrix.columns:
                if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold:
                    if col1 not in to_keep:
                        to_keep.add(col1)
                    if col2 not in to_keep and col2 not in to_drop:
                        to_drop.add(col2)
        
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} highly correlated columns: {list(to_drop)}")
            df = df.drop(list(to_drop))
        
        return df
    
    def _remove_sparse_columns(self, df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
        """Remove columns with high missing values"""
        logger.info(f"Removing columns with more than {(1-threshold)*100}% missing values...")
        
        total_rows = df.height
        null_counts = df.null_count()
        
        cols_to_keep = []
        for col in df.columns:
            null_count = null_counts[col][0]
            non_null_ratio = (total_rows - null_count) / total_rows
            if non_null_ratio >= threshold:
                cols_to_keep.append(col)
        
        removed_cols = set(df.columns) - set(cols_to_keep)
        if removed_cols:
            logger.info(f"Removed columns: {list(removed_cols)}")
            df = df.select(cols_to_keep)
        
        return df
