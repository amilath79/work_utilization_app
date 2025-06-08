import mlflow
import os

# Set tracking URI
mlflow.set_tracking_uri('http://localhost:5000')

print("=== MLFLOW RUN DIAGNOSTIC ===")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

try:
    # Set experiment
    mlflow.set_experiment('workforce_prediction')
    
    print(" Starting test run...")
    
    # Test basic run creation
    with mlflow.start_run(run_name='diagnostic_test_run') as run:
        print(f" Run started: {run.info.run_id}")
        
        # Log test metrics
        mlflow.log_metrics({
            'test_mae': 0.123,
            'test_rmse': 0.456
        })
        print(" Metrics logged")
        
        # Log test parameters
        mlflow.log_params({
            'test_param': 'test_value'
        })
        print(" Parameters logged")
        
        print(f" Run status: {run.info.status}")
    
    print(" Context manager completed")
    
    # Verify run was saved
    runs = mlflow.search_runs(experiment_ids=['1'])
    print(f" Runs found after test: {len(runs)}")
    
    if len(runs) > 0:
        print(" SUCCESS: Run was saved!")
        for i, row in runs.iterrows():
            run_name = row.get('tags.mlflow.runName', 'Unnamed')
            print(f"  - {run_name}")
    else:
        print(" FAILED: Run was not saved")
        
except Exception as e:
    print(f" ERROR: {e}")
    import traceback
    traceback.print_exc()
