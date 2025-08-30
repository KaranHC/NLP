import polars as pl
from datetime import date
from typing import Tuple
from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
    
    def engineer_basic_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Basic feature engineering without temporal leakage"""
        logger.info("Starting basic feature engineering...")
        
        # Drop unnecessary columns
        if "upc" in df.columns:
            df = df.drop("upc")
        
        # Drop rows with null values
        df = df.drop_nulls()
        
        # Territory risk scoring
        risk_expr = (
            pl.when(pl.col("territory").is_in(self.config.HIGH_RISK_TERRITORIES))
            .then(3)
            .when(pl.col("territory").is_in(['South Korea', 'Russia', 'Ukraine', 'Brazil']))
            .then(2)
            .when(pl.col("territory").is_in(['United States', 'United Kingdom', 'Canada', 'Australia']))
            .then(1)
            .otherwise(2)
        )
        
        df = df.with_columns(risk_expr.alias("territory_risk_score"))
        
        logger.info(f"Basic feature engineering complete. Shape: {df.shape}")
        return df
    
    def create_temporal_splits(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Create temporal splits with proper feature engineering"""
        logger.info("Creating temporal splits...")
        
        # Define split dates
        train_cutoff = date(2024, 10, 1)
        val_cutoff = date(2024, 11, 1)
        test_cutoff = date(2024, 12, 1)
        
        # Create training set
        train_df = self._create_train_features(df, train_cutoff)
        val_df = self._create_validation_features(df, val_cutoff)
        test_df = self._create_test_features(df, test_cutoff)
        
        # Remove statement_period column
        train_df = train_df.drop('statement_period')
        val_df = val_df.drop('statement_period')
        test_df = test_df.drop('statement_period')
        
        logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        return train_df, val_df, test_df
    
    def _create_train_features(self, df: pl.DataFrame, cutoff_date: date) -> pl.DataFrame:
        """Create training features"""
        train_periods = df.filter(pl.col('statement_period') <= cutoff_date)
        
        # Add artist-level features using window functions
        ctx = pl.SQLContext(train_data=train_periods)
        
        artist_features_query = """
        SELECT *,
            COUNT(DISTINCT account_id) OVER (PARTITION BY artist) AS connected_accounts,
            COUNT(DISTINCT payee_id) OVER (PARTITION BY artist) AS connected_payees,
            COUNT(DISTINCT territory) OVER (PARTITION BY artist) AS geographic_spread,
            SUM(CAST(quantity AS BIGINT)) OVER (PARTITION BY artist) AS total_artist_streams,
            AVG(CAST(gross_revenue_in_usd AS DOUBLE)) OVER (PARTITION BY artist) AS avg_artist_revenue,
            COUNT(DISTINCT "service") OVER (PARTITION BY artist) AS platform_diversity,
            SUM(
                CASE 
                    WHEN territory IN ('Turkey', 'Finland', 'Indonesia', 'India') 
                    THEN CAST(quantity AS BIGINT) 
                    ELSE 0 
                END
            ) OVER (PARTITION BY artist) AS high_risk_streams
        FROM train_data
        """
        
        train_df = ctx.execute(artist_features_query).collect()
        
        # Add derived features
        train_df = train_df.with_columns([
            pl.when(
                (pl.col("high_risk_streams").cast(pl.Float64) / 
                 pl.col("total_artist_streams").cast(pl.Float64).clip(lower_bound=1.0)) > 0.7
            ).then(pl.lit(3))
            .when(
                (pl.col("high_risk_streams").cast(pl.Float64) / 
                 pl.col("total_artist_streams").cast(pl.Float64).clip(lower_bound=1.0)) > 0.4
            ).then(pl.lit(2))
            .otherwise(pl.lit(1))
            .alias("artist_geographic_risk")
        ])
        
        # Add account-level features
        train_df = self._add_account_features(train_df)
        
        return train_df
    
    def _create_validation_features(self, df: pl.DataFrame, cutoff_date: date) -> pl.DataFrame:
        """Create validation features using historical data"""
        return self._create_temporal_features_no_leakage(df, cutoff_date)
    
    def _create_test_features(self, df: pl.DataFrame, cutoff_date: date) -> pl.DataFrame:
        """Create test features using historical data"""
        return self._create_temporal_features_no_leakage(df, cutoff_date)
    
    def _create_temporal_features_no_leakage(self, df: pl.DataFrame, cutoff_date: date) -> pl.DataFrame:
        """Create features using only historical data to prevent leakage"""
        historical_df = df.filter(pl.col('statement_period') < cutoff_date)
        current_df = df.filter(pl.col('statement_period') == cutoff_date)
        
        if historical_df.height > 0:
            ctx = pl.SQLContext()
            ctx.register("historical_data", historical_df)
            
            # Create artist features from historical data
            artist_query = """
            WITH artist_aggregates AS (
                SELECT 
                    artist,
                    COUNT(DISTINCT account_id) AS connected_accounts,
                    COUNT(DISTINCT payee_id) AS connected_payees,
                    COUNT(DISTINCT territory) AS geographic_spread,
                    SUM(CAST(quantity AS BIGINT)) AS total_artist_streams,
                    AVG(CAST(gross_revenue_in_usd AS DOUBLE)) AS avg_artist_revenue,
                    COUNT(DISTINCT "service") AS platform_diversity,
                    SUM(
                        CASE 
                            WHEN territory IN ('Turkey', 'Finland', 'Indonesia', 'India') 
                            THEN CAST(quantity AS BIGINT) 
                            ELSE 0 
                        END
                    ) AS high_risk_streams
                FROM historical_data
                GROUP BY artist
            )
            SELECT *,
                CAST(CASE
                    WHEN (CAST(high_risk_streams AS DOUBLE) / GREATEST(CAST(total_artist_streams AS DOUBLE), 1.0)) > 0.7 THEN 3
                    WHEN (CAST(high_risk_streams AS DOUBLE) / GREATEST(CAST(total_artist_streams AS DOUBLE), 1.0)) > 0.4 THEN 2
                    ELSE 1
                END AS INTEGER) AS artist_geographic_risk
            FROM artist_aggregates
            """
            
            artist_features = ctx.execute(artist_query).collect()
        else:
            # Create empty dataframe with proper schema
            schema = {
                "artist": pl.Utf8,
                "connected_accounts": pl.UInt32,
                "connected_payees": pl.UInt32,
                "geographic_spread": pl.UInt32,
                "total_artist_streams": pl.Int64,
                "avg_artist_revenue": pl.Float64,
                "platform_diversity": pl.UInt32,
                "high_risk_streams": pl.Int64,
                "artist_geographic_risk": pl.Int32
            }
            artist_features = pl.DataFrame(schema=schema)
        
        # Join features
        result_df = current_df.join(artist_features, on="artist", how="left")
        
        # Fill nulls
        fill_values = [
            pl.col("connected_accounts").fill_null(0),
            pl.col("connected_payees").fill_null(0),
            pl.col("geographic_spread").fill_null(0),
            pl.col("total_artist_streams").fill_null(0),
            pl.col("avg_artist_revenue").fill_null(0.0),
            pl.col("platform_diversity").fill_null(0),
            pl.col("high_risk_streams").fill_null(0),
            pl.col("artist_geographic_risk").fill_null(1)
        ]
        result_df = result_df.with_columns(fill_values)
        
        # Add account-level features
        result_df = self._create_account_level_features_no_leakage(df, result_df, cutoff_date)
        
        return result_df
    
    def _create_account_level_features_no_leakage(self, hs_df: pl.DataFrame, df: pl.DataFrame, cutoff_date: date) -> pl.DataFrame:
        """Create account-level features using historical data"""
        historical_df = hs_df.filter(pl.col('statement_period') < cutoff_date)
        
        if historical_df.height > 0:
            ctx = pl.SQLContext()
            ctx.register("historical_data", historical_df)
            
            query = """
            WITH global_revenue_stats AS (
                SELECT
                    AVG(CAST(gross_revenue_in_usd AS DOUBLE)) AS global_avg,
                    STDDEV(CAST(gross_revenue_in_usd AS DOUBLE)) AS global_stddev
                FROM historical_data
            ),
            account_quantity_sum AS (
                SELECT
                    account_id,
                    SUM(CAST(quantity AS BIGINT)) as sum_quantity
                FROM historical_data
                GROUP BY account_id
            ),
            territory_concentration AS (
                SELECT
                    historical_data.account_id,
                    SUM(POWER(CAST(h.quantity AS DOUBLE) / NULLIF(account_quantity_sum.sum_quantity, 0), 2)) AS territory_concentration_index
                FROM historical_data h
                JOIN account_quantity_sum ON historical_data.account_id = account_quantity_sum.account_id
                GROUP BY h.account_id
            ),
            account_metrics_agg AS (
                SELECT
                    account_id,
                    COUNT(*) AS total_transactions,
                    COUNT(DISTINCT territory) AS territory_diversity,
                    COUNT(DISTINCT "service") AS service_diversity,
                    COUNT(DISTINCT artist) AS artist_diversity,
                    SUM(CAST(quantity AS BIGINT)) AS total_streams,
                    AVG(CAST(gross_revenue_in_usd AS DOUBLE)) AS avg_revenue,
                    STDDEV(CAST(gross_revenue_in_usd AS DOUBLE)) AS revenue_stddev,
                    MAX(CAST(quantity AS BIGINT)) AS max_streams_single
                FROM historical_data
                GROUP BY account_id
            )
            SELECT
                account_metrics_agg.account_id,
                account_metrics_agg.total_transactions,
                account_metrics_agg.territory_diversity,
                account_metrics_agg.service_diversity,
                account_metrics_agg.artist_diversity,
                account_metrics_agg.total_streams,
                account_metrics_agg.avg_revenue,
                account_metrics_agg.revenue_stddev,
                account_metrics_agg.max_streams_single,
                territory_concentration.territory_concentration_index,
                CAST(CASE
                    WHEN account_metrics_agg.avg_revenue > (global_revenue_stats.global_avg + 3 * global_revenue_stats.global_stddev) THEN 1
                    ELSE 0
                END AS BIGINT) AS is_revenue_outlier,
                CAST(CASE
                    WHEN territory_concentration.territory_concentration_index > 0.8 THEN 3
                    WHEN territory_concentration.territory_concentration_index > 0.6 THEN 2
                    ELSE 1
                END AS INTEGER) AS concentration_risk_score
            FROM account_metrics_agg
            LEFT JOIN territory_concentration ON account_metrics_agg.account_id = territory_concentration.account_id
            CROSS JOIN global_revenue_stats
            """
            
            account_metrics = ctx.execute(query).collect()
        else:
            # Create empty schema
            schema = {
                "account_id": pl.Utf8,
                "total_transactions": pl.UInt32,
                "territory_diversity": pl.UInt32,
                "service_diversity": pl.UInt32,
                "artist_diversity": pl.UInt32,
                "total_streams": pl.Int64,
                "avg_revenue": pl.Float64,
                "revenue_stddev": pl.Float64,
                "max_streams_single": pl.Int64,
                "territory_concentration_index": pl.Float64,
                "is_revenue_outlier": pl.Int64,
                "concentration_risk_score": pl.Int32,
            }
            account_metrics = pl.DataFrame(schema=schema)
        
        # Join and fill nulls
        result_df = df.join(account_metrics, on="account_id", how="left")
        
        fill_values = [
            pl.col("total_transactions").fill_null(0),
            pl.col("territory_diversity").fill_null(0),
            pl.col("service_diversity").fill_null(0),
            pl.col("artist_diversity").fill_null(0),
            pl.col("total_streams").fill_null(0),
            pl.col("avg_revenue").fill_null(0.0),
            pl.col("revenue_stddev").fill_null(0.0),
            pl.col("max_streams_single").fill_null(0),
            pl.col("territory_concentration_index").fill_null(0.0),
            pl.col("is_revenue_outlier").fill_null(0),
            pl.col("concentration_risk_score").fill_null(1),
        ]
        result_df = result_df.with_columns(fill_values)
        
        return result_df
    
    def _add_account_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add account-level features to training data"""
        ctx = pl.SQLContext()
        ctx.register("train_data_final", df)
        
        account_features_query = """
        WITH 
        global_revenue_stats AS (
            SELECT
                AVG(CAST(gross_revenue_in_usd AS DOUBLE)) AS global_avg,
                STDDEV(CAST(gross_revenue_in_usd AS DOUBLE)) AS global_stddev
            FROM train_data_final
        ),
        account_quantity_sum AS (
            SELECT account_id, SUM(CAST(quantity AS DOUBLE)) as sum_quantity
            FROM train_data_final GROUP BY account_id
        ),
        territory_concentration AS (
            SELECT
                account_id,
                SUM(POWER(CAST(quantity AS DOUBLE) / NULLIF(sum_quantity, 0), 2)) AS territory_concentration_index
            FROM train_data_final
            JOIN account_quantity_sum USING(account_id)
            GROUP BY account_id
        ),
        account_metrics_agg AS (
            SELECT
                account_id,
                COUNT(*) AS total_transactions,
                COUNT(DISTINCT territory) AS territory_diversity,
                COUNT(DISTINCT "service") AS service_diversity,
                COUNT(DISTINCT artist) AS artist_diversity,
                SUM(CAST(quantity AS BIGINT)) AS total_streams,
                AVG(CAST(gross_revenue_in_usd AS DOUBLE)) AS avg_revenue,
                STDDEV(CAST(gross_revenue_in_usd AS DOUBLE)) AS revenue_stddev,
                MAX(CAST(quantity AS BIGINT)) AS max_streams_single
            FROM train_data_final
            GROUP BY account_id
        )
        SELECT
            account_metrics_agg.account_id,
            account_metrics_agg.total_transactions,
            account_metrics_agg.territory_diversity,
            account_metrics_agg.service_diversity,
            account_metrics_agg.artist_diversity,
            account_metrics_agg.total_streams,
            account_metrics_agg.avg_revenue,
            account_metrics_agg.revenue_stddev,
            account_metrics_agg.max_streams_single,
            territory_concentration.territory_concentration_index,
            CAST(CASE WHEN account_metrics_agg.avg_revenue > (global_revenue_stats.global_avg + 3 * global_revenue_stats.global_stddev) THEN 1 ELSE 0 END AS BIGINT) AS is_revenue_outlier,
            CAST(CASE WHEN territory_concentration.territory_concentration_index > 0.8 THEN 3 WHEN territory_concentration.territory_concentration_index > 0.6 THEN 2 ELSE 1 END AS INTEGER) AS concentration_risk_score
        FROM account_metrics_agg
        LEFT JOIN territory_concentration ON account_metrics_agg.account_id = territory_concentration.account_id
        CROSS JOIN global_revenue_stats
        """
        
        account_features = ctx.execute(account_features_query).collect()
        
        # Join account features
        result_df = df.join(account_features, on="account_id", how="left")
        
        # Fill nulls
        result_df = result_df.with_columns([
            pl.col("total_transactions").fill_null(0),
            pl.col("territory_diversity").fill_null(0),
            pl.col("service_diversity").fill_null(0),
            pl.col("artist_diversity").fill_null(0),
            pl.col("total_streams").fill_null(0),
            pl.col("avg_revenue").fill_null(0.0),
            pl.col("revenue_stddev").fill_null(0.0),
            pl.col("max_streams_single").fill_null(0),
            pl.col("territory_concentration_index").fill_null(0.0),
            pl.col("is_revenue_outlier").fill_null(0),
            pl.col("concentration_risk_score").fill_null(1)
        ])
        
        return result_df
