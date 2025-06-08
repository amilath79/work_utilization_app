"""
Enterprise MLflow Test - Complete Verification
"""
import sys
import os
sys.path.append('C:/forlogssystems/work_utilization_app')

from utils.enterprise_mlflow import mlflow_manager
from config import FEATURE_GROUPS, MLFLOW_ENABLE_TRACKING, MLFLOW_TRACKING_URI, MODELS_DIR, MLFLOW_EXPERIMENT_NAME
import mlflow

def test_enterprise_mlflow():
    print("🧪 TESTING ENTERPRISE MLFLOW - COMPLETE VERIFICATION")
    print("=" * 60)
    
    # 1. Project Structure Verification
    print("\n📁 PROJECT STRUCTURE:")
    print(f"   Project Directory: C:/forlogssystems/work_utilization_app")
    print(f"   Models Directory: {MODELS_DIR}")
    print(f"   MLflow Enabled: {MLFLOW_ENABLE_TRACKING}")
    print(f"   Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"   Experiment Name: {MLFLOW_EXPERIMENT_NAME}")
    
    # Verify directories exist
    if os.path.exists(MODELS_DIR):
        print(f"   ✅ Models directory exists")
    else:
        print(f"   ❌ Models directory missing: {MODELS_DIR}")
        
    # 2. Active Feature Groups
    active_groups = [k for k, v in FEATURE_GROUPS.items() if v]
    print(f"\n🎯 ACTIVE FEATURE GROUPS: {active_groups}")
    
    # 3. Server Connection Test
    print("\n🌐 SERVER CONNECTION TEST:")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"   🔗 Connecting to: {MLFLOW_TRACKING_URI}")
        
        # Test basic connection
        experiments = mlflow.search_experiments()
        print(f"   ✅ Server connection successful")
        print(f"   📊 Found {len(experiments)} total experiments")
        
    except Exception as e:
        print(f"   ❌ Server connection failed: {e}")
        print("   💡 Make sure MLflow server is running:")
        print("      mlflow server --backend-store-uri file:///C:/forlogssystems/Models/mlflow-runs --port 5000")
        return False
    
    # 4. List All Experiments
    print("\n📋 AVAILABLE EXPERIMENTS:")
    workforce_experiment = None
    
    for exp in experiments:
        print(f"   📁 {exp.name} (ID: {exp.experiment_id})")
        if exp.name == MLFLOW_EXPERIMENT_NAME:
            workforce_experiment = exp
            print(f"      ✅ Found target experiment!")
            
    # 5. Check Existing Runs in workforce_prediction
    if workforce_experiment:
        print(f"\n📊 CHECKING RUNS IN '{MLFLOW_EXPERIMENT_NAME}' EXPERIMENT:")
        try:
            existing_runs = mlflow.search_runs(experiment_ids=[workforce_experiment.experiment_id])
            print(f"   📈 Total runs found: {len(existing_runs)}")
            
            if len(existing_runs) > 0:
                print("   🎯 RECENT RUNS:")
                for i, run in existing_runs.head(10).iterrows():
                    run_name = run.get('tags.mlflow.runName', 'Unnamed')
                    status = run.get('status', 'Unknown')
                    start_time = run.get('start_time', 'Unknown')
                    print(f"      📌 {run_name} | Status: {status}")
                    
                    # Show metrics if available
                    metrics_cols = [col for col in existing_runs.columns if col.startswith('metrics.')]
                    if metrics_cols:
                        print(f"         Metrics: {metrics_cols}")
                        
                print(f"\n   🌐 VIEW IN BROWSER:")
                print(f"      1. Go to: {MLFLOW_TRACKING_URI}")
                print(f"      2. Click on '{MLFLOW_EXPERIMENT_NAME}' in left panel")
                print(f"      3. You should see {len(existing_runs)} runs")
                
            else:
                print("   ⚠️  No runs found - this is why MLflow UI appears empty")
                
        except Exception as e:
            print(f"   ❌ Error checking runs: {e}")
    else:
        print(f"\n⚠️  '{MLFLOW_EXPERIMENT_NAME}' experiment not found")
    
    # 6. MLflow Manager Test
    print(f"\n🔧 MLFLOW MANAGER TEST:")
    success = mlflow_manager.initialize()
    print(f"   Manager Initialization: {'✅' if success else '❌'}")
    
    if success:
        # 7. Create Test Run
        print("\n🧪 CREATING TEST RUN:")
        try:
            with mlflow_manager.start_run("verification_test_run") as run:
                if run:
                    print(f"   ✅ Test run created: {run.info.run_id}")
                    
                    # Log test metrics
                    test_metrics = {
                        "MAE": 0.123, 
                        "RMSE": 0.456,
                        "R²": 0.85,
                        "MAPE": 8.5
                    }
                    
                    mlflow_manager.log_model_metrics(
                        work_type="TEST_209",
                        metrics=test_metrics
                    )
                    
                    print(f"   📊 Logged test metrics: {list(test_metrics.keys())}")
                    print(f"   🎯 Experiment ID: {run.info.experiment_id}")
                    
                    # Browser instructions
                    print(f"\n🌐 VIEW TEST RUN IN BROWSER:")
                    print(f"   1. Open: {MLFLOW_TRACKING_URI}")
                    print(f"   2. Click '{MLFLOW_EXPERIMENT_NAME}' in left panel")
                    print(f"   3. Look for run: 'verification_test_run'")
                    print(f"   4. Run ID: {run.info.run_id}")
                    
                    return True
                else:
                    print("   ❌ Failed to create test run")
                    return False
                    
        except Exception as e:
            print(f"   ❌ Error creating test run: {e}")
            return False
    
    return False

def check_previous_training_runs():
    """Check for runs from previous training sessions"""
    print("\n🔍 CHECKING FOR PREVIOUS TRAINING RUNS:")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Search for training session runs
        all_runs = mlflow.search_runs(experiment_ids=None)
        
        training_runs = []
        for i, run in all_runs.iterrows():
            run_name = run.get('tags.mlflow.runName', '')
            if 'training_session' in run_name or 'model_' in run_name:
                training_runs.append({
                    'name': run_name,
                    'run_id': run.get('run_id', ''),
                    'status': run.get('status', ''),
                    'start_time': run.get('start_time', '')
                })
        
        if training_runs:
            print(f"   📈 Found {len(training_runs)} training-related runs:")
            for run in training_runs[:10]:  # Show first 10
                print(f"      📌 {run['name']} | {run['status']}")
        else:
            print("   ⚠️  No training runs found")
            
    except Exception as e:
        print(f"   ❌ Error checking training runs: {e}")

if __name__ == "__main__":
    # Run main test
    success = test_enterprise_mlflow()
    
    # Check for previous training runs
    check_previous_training_runs()
    
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 MLFLOW TEST COMPLETED SUCCESSFULLY!")
        print("💡 If you still don't see data in browser:")
        print("   1. Refresh the browser page")
        print("   2. Make sure you click 'workforce_prediction' experiment")
        print("   3. Check that MLflow server is still running")
    else:
        print("❌ MLFLOW TEST FAILED")
        print("💡 Check the error messages above and fix issues")
    
    print("\n🌐 MLflow UI: http://localhost:5000/")
    print("=" * 60)