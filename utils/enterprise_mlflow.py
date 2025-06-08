"""
Simple Enterprise MLflow Manager
"""
import mlflow
import mlflow.sklearn
import logging
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager

# Import from current config structure
try:
    from config import (
        MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_ENABLE_TRACKING,
        ENTERPRISE_CONFIG, enterprise_logger, audit_logger
    )
except ImportError:
    # Fallback if enterprise config not available
    MLFLOW_TRACKING_URI = "file:///C:/forlogssystems/mlflow-runs"
    MLFLOW_EXPERIMENT_NAME = "workforce_prediction"
    MLFLOW_ENABLE_TRACKING = True
    enterprise_logger = logging.getLogger(__name__)
    audit_logger = logging.getLogger(__name__)
    
    class MockConfig:
        class Environment:
            value = "development"
        environment = Environment()
    
    ENTERPRISE_CONFIG = MockConfig()

class EnterpriseMLflowManager:
    """Simple Enterprise MLflow manager"""
    
    def __init__(self):
        self.logger = enterprise_logger
        self.enabled = MLFLOW_ENABLE_TRACKING
        self.initialized = False
    
    def initialize(self) -> bool:
        """Simple MLflow initialization"""
        if not self.enabled:
            self.logger.info("MLflow tracking disabled")
            return False
            
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            # Create experiment if not exists
            try:
                experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
                if experiment is None:
                    mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            except:
                mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            self.initialized = True
            
            self.logger.info(f"✅ MLflow initialized: {MLFLOW_EXPERIMENT_NAME}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MLflow initialization failed: {e}")
            return False
    
    @contextmanager
    def start_run(self, run_name: str, nested: bool = False, tags: Optional[Dict[str, str]] = None):
        if not self.initialized:
            self.logger.warning("MLflow not initialized, skipping run tracking")
            yield None
            return
        
        run = None
        try:
            # Add enterprise tags
            enterprise_tags = {
                "environment": "production",  # Simplified
                "timestamp": datetime.now().isoformat()
            }
            
            if tags:
                enterprise_tags.update(tags)
            
            run = mlflow.start_run(run_name=run_name, nested=nested, tags=enterprise_tags)
            self.logger.info(f"Started MLflow run: {run_name} | Run ID: {run.info.run_id}")
            
            yield run
            
        except Exception as e:
            self.logger.error(f"Error in MLflow run {run_name}: {e}")
            raise
        finally:
            if run:
                try:
                    # ✅ EXPLICIT SAVE AND END
                    mlflow.end_run(status='FINISHED')  # Ensure proper status
                    self.logger.info(f"Ended and saved MLflow run: {run_name} | Run ID: {run.info.run_id}")
                except Exception as e:
                    self.logger.error(f"Error ending MLflow run: {e}")
    
    def log_model_metrics(self, work_type: str, metrics: Dict[str, float], 
                         cv_scores: Optional[Dict[str, list]] = None) -> None:
        """Log model performance metrics"""
        try:
            # Log primary metrics
            mlflow.log_metrics(metrics)
            
            # Log cross-validation scores if available
            if cv_scores:
                import pandas as pd
                for metric_name, scores in cv_scores.items():
                    mlflow.log_metrics({
                        f"{metric_name}_mean": float(pd.Series(scores).mean()),
                        f"{metric_name}_std": float(pd.Series(scores).std())
                    })
            
            self.logger.info(f"Logged metrics for work type {work_type}: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics for {work_type}: {e}")
    
    def log_model_artifact(self, model: Any, work_type: str, 
                          feature_importance: Optional[Dict[str, float]] = None) -> None:
        """Log model artifacts"""
        try:
            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_{work_type}"
            )
            
            # Log feature importance as artifact
            if feature_importance:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(feature_importance, f, indent=2)
                    f.flush()
                    mlflow.log_artifact(f.name, f"feature_importance_{work_type}.json")
                    os.unlink(f.name)
            
            self.logger.info(f"Logged model artifacts for work type {work_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to log model artifacts for {work_type}: {e}")
    
    def log_training_parameters(self, params: Dict[str, Any]) -> None:
        """Log training parameters"""
        try:
            # Sanitize parameters
            sanitized_params = {}
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    sanitized_params[key] = value
                elif isinstance(value, (list, tuple)):
                    sanitized_params[key] = str(value)
                else:
                    sanitized_params[key] = str(type(value))
            
            mlflow.log_params(sanitized_params)
            self.logger.info(f"Logged training parameters: {list(sanitized_params.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")
    
    def cleanup(self) -> None:
        """Cleanup MLflow resources"""
        try:
            if self.initialized:
                self.logger.info("Cleaning up MLflow resources")
        except Exception as e:
            self.logger.error(f"Error during MLflow cleanup: {e}")

# Global enterprise MLflow manager instance
mlflow_manager = EnterpriseMLflowManager()