import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, recall_score, matthews_corrcoef, confusion_matrix
)
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class ModelEvaluator:
    def __init__(self):
        pass
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        logger.info("Evaluating model performance...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'auc_pr': average_precision_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }
        
        # Log metrics
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return metrics
