import mlflow
import json
from mlflow.models.signature import infer_signature
from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class ModelDeployer:
    def __init__(self, config: Config):
        self.config = config
    
    def deploy_model(self, model, X_test, y_test, cat_features, best_params, 
                    model_stage: str = "Staging", register_model: bool = True):
        """Deploy model to MLflow Model Registry"""
        logger.info("Deploying model...")
        
        # Make predictions for signature
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Log metrics and parameters
        mlflow.log_params({f"final_{k}": v for k, v in best_params.items()})
        mlflow.log_text("\n".join(cat_features), "categorical_features.txt")
        mlflow.log_text(json.dumps({"columns": X_test.columns.tolist()}), "feature_columns.json")
        
        # Model signature and input example
        signature = infer_signature(X_test, y_pred_proba)
        input_example = X_test.head(3)
        
        # Register model
        registered_model_name = self.config.model_registry_name if register_model else None
        
        model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name
        )
        
        # Set model alias
        if register_model and model_stage:
            client = mlflow.tracking.MlflowClient()
            model_version = model_info.registered_model_version
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=model_stage,
                version=model_version
            )
        
        logger.info(f"Model deployed successfully. Registry name: {registered_model_name}")
        return model_info
