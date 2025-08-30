import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    """Configuration class for the fraud detection pipeline"""
    # Data paths
    CATALOG_NAME: str = os.getenv("CATALOG_NAME", "fraud_data")
    SCHEMA_NAME: str = os.getenv("SCHEMA_NAME", "fraud_account")
    RAW_TABLE_NAME: str = "raw_complete_fraud_data_snowflake_snapshot"
    
    # Model configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "catboost_binary_classifier")
    EXPERIMENT_NAME: str = os.getenv("EXPERIMENT_NAME", "/fraud_detection_experiment")
    
    # Training parameters
    RANDOM_SEED: int = 42
    CV_FOLDS: int = 5
    OPTUNA_TRIALS: int = 10
    TEST_SIZE: float = 0.2
    
    # High risk territories
    HIGH_RISK_TERRITORIES: List[str] = None
    
    def __post_init__(self):
        if self.HIGH_RISK_TERRITORIES is None:
            self.HIGH_RISK_TERRITORIES = ['Turkey', 'Finland', 'Indonesia', 'India', 'Cyprus']
    
    @property
    def full_table_name(self) -> str:
        return f"{self.CATALOG_NAME}.{self.SCHEMA_NAME}.{self.RAW_TABLE_NAME}"
    
    @property
    def model_registry_name(self) -> str:
        return f"{self.CATALOG_NAME}.{self.SCHEMA_NAME}.{self.MODEL_NAME}"
