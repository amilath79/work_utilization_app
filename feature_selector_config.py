# Feature Selection Configuration
# Modify these parameters to customize your feature selection process

# Data Configuration
DATA_PATH = "data/processed/workforce_data.csv"
TARGET_PUNCH_CODES = [202, 203, 206, 209, 210, 211, 213, 214, 215, 217]

# Feature Parameter Variations to Test
LAG_VARIATIONS = [
    {'lags': [1, 7]},
    {'lags': [1, 7, 14]},
    {'lags': [1, 2, 3, 7, 14]},
    {'lags': [1, 2, 3, 7, 14, 21, 28]}
]

ROLLING_VARIATIONS = [
    {'windows': [7]},
    {'windows': [7, 14]},
    {'windows': [3, 7, 14]},
    {'windows': [3, 7, 14, 30]}
]

# Model Configuration (matching train_models2.py)
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Cross-validation Configuration
CV_SPLITS = 3

# Performance Thresholds (for filtering results)
TARGET_MAE = 0.5
TARGET_MAPE = 10.0
TARGET_R2 = 0.85

# Feature Selection Strategy
# Options: 'comprehensive', 'quick', 'custom'
SELECTION_STRATEGY = 'comprehensive'

# Quick mode: Test only most promising combinations
QUICK_MODE_CONFIGS = [
    # Individual features
    {'LAG_FEATURES': True, 'lag_params': {'lags': [1, 7, 14]}},
    {'ROLLING_FEATURES': True, 'rolling_params': {'windows': [7, 14]}},
    {'DATE_FEATURES': True},
    {'CYCLICAL_FEATURES': True},
    
    # Best combinations
    {'LAG_FEATURES': True, 'ROLLING_FEATURES': True, 'DATE_FEATURES': True,
     'lag_params': {'lags': [1, 7, 14]}, 'rolling_params': {'windows': [7, 14]}},
    
    {'LAG_FEATURES': True, 'CYCLICAL_FEATURES': True, 'TREND_FEATURES': True,
     'lag_params': {'lags': [1, 2, 3, 7, 14]}},
]

# Logging Configuration
LOG_LEVEL = 'INFO'
SAVE_DETAILED_RESULTS = True
SAVE_INTERMEDIATE_RESULTS = True

# Output Configuration
RESULTS_DIR = 'feature_selection_results'
GENERATE_PLOTS = True
EXPORT_BEST_CONFIG = True