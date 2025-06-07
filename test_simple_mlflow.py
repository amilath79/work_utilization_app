"""
Enterprise MLflow Test
"""
import sys
sys.path.append('.')

from utils.enterprise_mlflow import mlflow_manager
from config import FEATURE_GROUPS, MLFLOW_ENABLE_TRACKING

def test_enterprise_mlflow():
    print("🧪 TESTING ENTERPRISE MLFLOW")
    print(f"MLflow Enabled: {MLFLOW_ENABLE_TRACKING}")
    print(f"Current Feature Groups: {[k for k, v in FEATURE_GROUPS.items() if v]}")
    
    # Test initialization
    success = mlflow_manager.initialize()
    print(f"Initialization: {'✅' if success else '❌'}")
    
    if success:
        # Test simple run
        with mlflow_manager.start_run("test_run") as run:
            if run:
                mlflow_manager.log_model_metrics(
                    work_type="TEST",
                    metrics={"MAE": 0.123, "R²": 0.85}
                )
                print("✅ Test run completed")
            else:
                print("❌ Test run failed")

if __name__ == "__main__":
    test_enterprise_mlflow()