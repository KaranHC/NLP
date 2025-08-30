import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from catboost import CatBoostClassifier, Pool
from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class HyperparameterOptimizer:
    def __init__(self, config: Config):
        self.config = config
        self.skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    
    def optimize(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                X_test: pd.DataFrame, y_test: np.ndarray, 
                cat_features: List[str]) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna with MLflow tracking"""
        logger.info("Starting hyperparameter optimization...")
        
        cat_idx = [X_train.columns.get_loc(c) for c in cat_features]
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "iterations": trial.suggest_int("iterations", 500, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "random_seed": self.config.RANDOM_SEED,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": False,
            }
            
            cv_scores = []
            with mlflow.start_run(nested=True, tags={"phase": "optuna_trial"}):
                mlflow.log_params(params)
                
                for fold, (train_idx, val_idx) in enumerate(self.skf.split(X_train, y_train), start=1):
                    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_va = y_train[train_idx], y_train[val_idx]
                    
                    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
                    val_pool = Pool(X_va, y_va, cat_features=cat_idx)
                    
                    model = CatBoostClassifier(**params)
                    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
                    
                    y_va_proba = model.predict_proba(X_va)[:, 1]
                    auc_score = roc_auc_score(y_va, y_va_proba)
                    cv_scores.append(auc_score)
                    
                    mlflow.log_metric(f"fold{fold}_auc", float(auc_score))
                
                mean_auc = float(np.mean(cv_scores))
                std_auc = float(np.std(cv_scores))
                mlflow.log_metric("cv_mean_auc", mean_auc)
                mlflow.log_metric("cv_std_auc", std_auc)
                
                return mean_auc
        
        # Start parent HPO run
        with mlflow.start_run(run_name="HPO", tags={"phase": "hpo"}):
            mlflow.set_tags({
                "framework": "catboost",
                "task": "binary_classification",
                "cv_scheme": f"StratifiedKFold(n_splits={self.config.CV_FOLDS}, shuffle=True)"
            })
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.config.OPTUNA_TRIALS)
            
            # Log best results
            best_params = study.best_params.copy()
            best_params.update({
                "random_seed": self.config.RANDOM_SEED, 
                "loss_function": "Logloss", 
                "verbose": False
            })
            
            mlflow.log_metric("best_cv_auc", float(study.best_value))
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            
            # Final model evaluation
            final_model = CatBoostClassifier(**best_params)
            final_model.fit(Pool(X_train, y_train, cat_features=cat_idx), verbose=False)
            
            y_test_proba = final_model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_test_proba)
            
            mlflow.log_metric("test_auc", float(test_auc))
            
            logger.info(f"Best CV AUC: {study.best_value:.4f}")
            logger.info(f"Test AUC: {test_auc:.4f}")
            
            return {
                "best_params": study.best_params,
                "best_cv_auc": float(study.best_value),
                "test_auc": float(test_auc),
                "n_trials": len(study.trials),
                "final_model": final_model
            }
