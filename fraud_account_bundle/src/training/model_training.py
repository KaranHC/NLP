import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import polars as pl
from catboost import CatBoostClassifier, Pool
from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config
    
    def prepare_features_for_training(self, train_df: pl.DataFrame, 
                                    test_df: pl.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str]]:
        """Prepare features for ML training"""
        logger.info("Preparing features for training...")
        
        # Extract target variable
        y_train = train_df.select(pl.col("fraud_type")).to_series()
        y_test = test_df.select(pl.col("fraud_type")).to_series()
        
        # Create feature matrices
        X_train = train_df.drop("fraud_type")
        X_test = test_df.drop("fraud_type")
        
        # Convert to binary classification (fraud_type==1 -> 1, else 0)
        y_train = pl.Series(np.where(y_train == 1, 1, 0))
        y_test = pl.Series(np.where(y_test == 1, 1, 0))
        
        # Convert to pandas for CatBoost compatibility
        X_train_pd = X_train.to_pandas()
        X_test_pd = X_test.to_pandas()
        
        # Convert target to numpy arrays
        y_train_np = y_train.to_numpy()
        y_test_np = y_test.to_numpy()
        
        # Identify categorical features
        cat_features = [col for col in X_train.columns if X_train[col].dtype == pl.Utf8]
        
        logger.info(f"Prepared {X_train_pd.shape[1]} features for training")
        logger.info(f"Target distribution - Train: {np.bincount(y_train_np)}")
        logger.info(f"Target distribution - Test: {np.bincount(y_test_np)}")
        
        return X_train_pd, y_train_np, X_test_pd, y_test_np, cat_features
    
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray,
                   X_test: pd.DataFrame, y_test: np.ndarray,
                   cat_features: List[str], best_params: Dict[str, Any]) -> CatBoostClassifier:
        """Train final model with best parameters"""
        logger.info("Training final model...")
        
        # Create pool objects
        cat_idx = [X_train.columns.get_loc(c) for c in cat_features]
        train_pool = Pool(X_train, y_train, cat_features=cat_idx)
        test_pool = Pool(X_test, y_test, cat_features=cat_idx)
        
        # Train model
        model = CatBoostClassifier(
            **best_params,
            random_seed=self.config.RANDOM_SEED,
            verbose=False
        )
        model.fit(train_pool, eval_set=test_pool, plot=False)
        
        logger.info("Model training complete")
        return model
