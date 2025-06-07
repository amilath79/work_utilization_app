"""
Enterprise-Grade MLflow Management
Handles experiment tracking with enterprise security and compliance
"""
import mlflow
import mlflow.sklearn
import logging
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
import pandas as pd

from config import MLFLOW_CONFIG, ENTERPRISE_CONFIG, enterprise_logger, audit_logger

class EnterpriseMLflowManager:
    """Enterprise-grade MLflow experiment tracking manager"""
    
    def __init__(self):
        self.logger = enterprise_logger
        self.audit_logger = audit_logger
        self.config = MLFLOW_CONFIG
        self.is_initialized = False
        self.current_experiment = None
        
    def initialize(self) -> bool:
        """Initialize MLflow with enterprise configuration"""
        if not self.config.enable_tracking:
            self.logger.info("MLflow tracking disabled in configuration")
            return False
            
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            self.logger.info(f"MLflow tracking URI set: {self.config.tracking_uri}")
            
            # Configure authentication if provided
            if self.config.username and self.config.password:
                os.environ['MLFLOW_TRACKING_USERNAME'] = self.config.username
                os.environ['MLFLOW_TRACKING_PASSWORD'] = self.config.password
                self.logger.info("MLflow authentication configured")
            
            if self.config.token:
                os.environ['MLFLOW_TRACKING_TOKEN'] = self.config.token
                self.logger.info("MLflow token authentication configured")
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(
                        self.config.experiment_name,
                        artifact_location=self.config.artifact_root
                    )
                    self.logger.info(f"Created new MLflow experiment: {self.config.experiment_name}")
                    self.audit_logger.info(f"EXPERIMENT_CREATED | {self.config.experiment_name} | {experiment_id}")
                else:
                    self.logger.info(f"Using existing MLflow experiment: {self.config.experiment_name}")
            except Exception as e:
                self.logger.error(f"Failed to create/get experiment: {e}")
                return False
            
            mlflow.set_experiment(self.config.experiment_name)
            self.current_experiment = self.config.experiment_name
            
            # Enable auto-logging if configured
            if self.config.auto_log:
                mlflow.sklearn.autolog(
                    log_input_examples=False,  # Don't log data for security
                    log_model_signatures=True,
                    log_models=True,
                    disable=False,
                    exclusive=False,
                    disable_for_unsupported_versions=False,
                    silent=False
                )
                self.logger.info("MLflow auto-logging enabled")
            
            self.is_initialized = True
            self.audit_logger.info(f"MLFLOW_INITIALIZED | {self.config.experiment_name} | SUCCESS")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow: {e}")
            self.audit_logger.error(f"MLFLOW_INIT_FAILED | {str(e)}")
            return False
    
    @contextmanager
    def start_run(self, run_name: str, nested: bool = False, tags: Optional[Dict[str, str]] = None):
        """Enterprise context manager for MLflow runs"""
        if not self.is_initialized:
            self.logger.warning("MLflow not initialized, skipping run tracking")
            yield None
            return
        
        run = None
        try:
            # Add enterprise tags
            enterprise_tags = {
                "environment": ENTERPRISE_CONFIG.environment.value,
                "enterprise_mode": str(ENTERPRISE_CONFIG.enterprise_mode),
                "timestamp": datetime.now().isoformat()
            }
            
            if tags:
                enterprise_tags.update(tags)
            
            run = mlflow.start_run(run_name=run_name, nested=nested, tags=enterprise_tags)
            self.logger.info(f"Started MLflow run: {run_name}")
            self.audit_logger.info(f"RUN_STARTED | {run_name} | {run.info.run_id}")
            
            yield run
            
        except Exception as e:
            self.logger.error(f"Error in MLflow run {run_name}: {e}")
            self.audit_logger.error(f"RUN_ERROR | {run_name} | {str(e)}")
            raise
        finally:
            if run:
                try:
                    mlflow.end_run()
                    self.logger.info(f"Ended MLflow run: {run_name}")
                    self.audit_logger.info(f"RUN_ENDED | {run_name} | {run.info.run_id}")
                except Exception as e:
                    self.logger.error(f"Error ending MLflow run: {e}")
    
    def log_model_metrics(self, work_type: str, metrics: Dict[str, float], 
                         cv_scores: Optional[Dict[str, list]] = None) -> None:
        """Log model performance metrics with enterprise compliance"""
        try:
            # Log primary metrics
            mlflow.log_metrics(metrics)
            
            # Log cross-validation scores if available
            if cv_scores:
                for metric_name, scores in cv_scores.items():
                    mlflow.log_metrics({
                        f"{metric_name}_mean": float(pd.Series(scores).mean()),
                        f"{metric_name}_std": float(pd.Series(scores).std()),
                        f"{metric_name}_min": float(pd.Series(scores).min()),
                        f"{metric_name}_max": float(pd.Series(scores).max())
                    })
            
            self.logger.info(f"Logged metrics for work type {work_type}: {metrics}")
            self.audit_logger.info(f"METRICS_LOGGED | {work_type} | {json.dumps(metrics)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics for {work_type}: {e}")
    
    def log_model_artifact(self, model: Any, work_type: str, 
                          feature_importance: Optional[Dict[str, float]] = None) -> None:
        """Log model artifacts with enterprise security considerations"""
        try:
            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_{work_type}",
                registered_model_name=f"workforce_model_{work_type}" if ENTERPRISE_CONFIG.enterprise_mode else None
            )
            
            # Log feature importance as artifact (not as data)
            if feature_importance:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(feature_importance, f, indent=2)
                    f.flush()
                    mlflow.log_artifact(f.name, f"feature_importance_{work_type}.json")
                    os.unlink(f.name)  # Clean up temp file
            
            self.logger.info(f"Logged model artifacts for work type {work_type}")
            self.audit_logger.info(f"MODEL_LOGGED | {work_type} | SUCCESS")
            
        except Exception as e:
            self.logger.error(f"Failed to log model artifacts for {work_type}: {e}")
            self.audit_logger.error(f"MODEL_LOG_FAILED | {work_type} | {str(e)}")
    
    def log_training_parameters(self, params: Dict[str, Any]) -> None:
        """Log training parameters with data sanitization"""
        try:
            # Sanitize parameters (remove sensitive data)
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
    
    def get_experiment_runs(self, experiment_name: Optional[str] = None) -> pd.DataFrame:
        """Get experiment runs with enterprise filtering"""
        try:
            exp_name = experiment_name or self.config.experiment_name
            experiment = mlflow.get_experiment_by_name(exp_name)
            
            if experiment is None:
                self.logger.warning(f"Experiment {exp_name} not found")
                return pd.DataFrame()
            
            runs = mlflow.search_runs(experiment.experiment_id)
            self.logger.info(f"Retrieved {len(runs)} runs from experiment {exp_name}")
            return runs
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment runs: {e}")
            return pd.DataFrame()
    
    def cleanup(self) -> None:
        """Cleanup MLflow resources"""
        try:
            if self.is_initialized:
                self.logger.info("Cleaning up MLflow resources")
                self.audit_logger.info("MLFLOW_CLEANUP | SUCCESS")
        except Exception as e:
            self.logger.error(f"Error during MLflow cleanup: {e}")

# Global enterprise MLflow manager instance
mlflow_manager = EnterpriseMLflowManager()