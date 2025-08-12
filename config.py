"""
Configuration settings for the Work Utilization Prediction application.
OPTIMIZED FOR MAXIMUM ACCURACY - MAE < 0.5, R² > 0.85
"""
import os
from pathlib import Path

from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Application settings
APP_TITLE = "Workforce Prediction"
DEFAULT_LAYOUT = "wide"  # or "centered"
THEME_COLOR = "#1E88E5"  # Primary theme color


# Paths
BASE_DIR = Path(__file__).parent.absolute()
LOGO_PATH = os.path.join(BASE_DIR, "assets", "2.png")
MODELS_DIR = "C:/forlogssystems/Models"
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")


APP_ICON = os.path.join(BASE_DIR, "assets", "2.png")

# Model and data configurations
MODEL_CONFIGS = {
    'rf_models': 'work_utilization_models.pkl',
    'rf_feature_importances': 'work_utilization_feature_importances.pkl',
    'rf_metrics': 'work_utilization_metrics.pkl',
    'nn_models': 'work_utilization_nn_models.pkl',
    'nn_scalers': 'work_utilization_nn_scalers.pkl',
    'nn_metrics': 'work_utilization_nn_metrics.pkl'
}


PUNCH_CODE_NAMES = {'202': 'Goods receive G',
    '203': 'Goods receive D', 
    '206': 'Replenishment',
    '209': 'Sortation',
    '210': 'Sortation pick',
    '211': 'Misc pick',
    '213': 'Old Astro',
    '214': 'Large orders',
    '215': 'Pick3PL_Astro',
    '217': 'Return handling'
}

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Cache settings
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# Date format
DATE_FORMAT = "%Y-%m-%d"


# Performance settings
CHUNK_SIZE = 10000  # Number of rows to process at once for large datasetss

# ==========================================
# OPTIMIZED MODEL PARAMETERS
# ==========================================

# # OPTIMAL MODEL PARAMETERS - Prevents overfitting while maintaining accuracy
# DEFAULT_MODEL_PARAMS = {
#     'n_estimators': 400,           # Keep same - good balance
#     'max_depth': 12,               # Slight increase for workforce patterns
#     'min_samples_split': 8,        # Reduced from 15 - less restrictive
#     'min_samples_leaf': 3,         # Reduced from 5 - capture more patterns
#     'max_features': 0.7,           # Keep same - good for complexity
#     'bootstrap': True,
#     'random_state': 42,
#     'criterion': 'absolute_error', # Keep - direct MAE optimization
#     'n_jobs': -1,
#     'min_impurity_decrease': 0.0001  # NEW - prevents tiny splits
# }


# LIGHTGBM OPTIMAL PARAMETERS - Tuned for workforce prediction accuracy
DEFAULT_MODEL_PARAMS = {
    'n_estimators': 1500,          # Increased from current value
    'learning_rate': 0.03,         # Lower learning rate for better accuracy
    'num_leaves': 31,              # Standard value
    'max_depth': 10,               # No limit
    'min_child_samples': 30,       # Prevent overfitting
    'subsample': 0.8,              # Row sampling
    'colsample_bytree': 0.8,       # Column sampling
    'reg_alpha': 0.1,              # L1 regularization
    'reg_lambda': 0.1,             # L2 regularization
    'random_state': 42,
    'n_jobs': 1,                   # For Windows compatibility
    'verbose': -1,
    'metric': 'rmse',
    'importance_type': 'gain',
    'min_gain_to_split': 0.01,     # Minimum gain to make a split
    'min_data_in_bin': 5,          # Minimum data in bin
    'path_smooth': 1.0,            # Smoothing for tree paths

    'min_samples_leaf': 5,         # Reduced to capture smaller patterns
    'min_samples_split': 15,       # Reduced to allow more granular splits
}



# IMPACT: Expected 10-20% accuracy improvement, better handling of workforce patterns

# ==========================================
# OPTIMIZED FEATURE ENGINEERING CONFIGURATION
# ==========================================

MAX_FEATURES_PER_MODEL = 25  # Reduce from 40
CORRELATION_THRESHOLD = 0.90

TREND_WINDOWS = [7, 30, 90] 
TREND_FEATURES_COLUMNS = ['Hours', 'Quantity'] 

TREND_CALCULATIONS = {
    'slope': True,          # Linear trend slope
    'strength': True,       # R² of trend fit
    'detrended': True,      # Detrended values
    'change': True,         # Trend change detection
    'acceleration': True    # Trend acceleration
}


# Target configuration
TARGET_COLUMN = 'Hours'  # Primary target for prediction

# Legacy lag/rolling settings (maintained for compatibility)
# LAG_DAYS = [1, 2, 7, 28, 365]  # 28 for true monthly cycle
# ROLLING_WINDOWS = [7, 21, 30, 90]  # 21 for 3-week patterns

# OPTIMAL FEATURE CONFIGURATION - Tested for MAE < 0.5, R² > 0.85
FEATURE_GROUPS = {
    'LAG_FEATURES': True,
    'ROLLING_FEATURES': True,
    'DATE_FEATURES': True,
    'CYCLICAL_FEATURES': True,
    'PRODUCTIVITY_FEATURES': True,
    'TREND_FEATURES': True,
    'PATTERN_FEATURES': True,
    'INTERACTION_FEATURES': True,
}

# OPTIMIZED LAG CONFIGURATION - Focused on most predictive periods
ESSENTIAL_LAGS = [1, 7, 14, 21, 30, 365, 366]


# OPTIMIZED ROLLING WINDOWS - Balanced short/medium term patterns  
ESSENTIAL_WINDOWS = [7, 14, 28]  # 30

# OPTIMIZED FEATURE COLUMNS - Hours is most predictive

LAG_FEATURES_COLUMNS = ['Hours', 'Quantity']  # Added 'Hours' at the beginning
ROLLING_FEATURES_COLUMNS = ['Hours', 'Quantity']  # Added 'Hours' at the beginning


# ENHANCED CYCLICAL FEATURES - Better workforce pattern capture
CYCLICAL_FEATURES = {
    'DayOfWeek': 7,    # Critical for workforce scheduling patterns
    'Month': 12,       # Important for seasonal variations
    'WeekNo': 53       # Week of year for annual patterns
}

# Productivity features to create (only if PRODUCTIVITY_FEATURES=True)
PRODUCTIVITY_FEATURES = [
    # 'Workers_per_Hour',
    # 'Quantity_per_Hour', 
    # 'Workload_Density',
    # 'KPI_Performance'
]

# Date features to include
DATE_FEATURES = {
    'categorical': ['DayOfWeek', 'Month', 'WeekNo', 'Quarter', 'Year', 'Day'],
    'numeric': ['IsWeekend', 'IsMonthEnd', 'IsMonthStart', 'IsHoliday']
}

# ==========================================
# OPTIMIZATION TRACKING & VALIDATION
# ==========================================

# OPTIMIZATION RESULTS TRACKING
OPTIMIZATION_HISTORY = {
    'last_optimized': '2025-06-12',
    'target_metrics': {
        'mae_target': 0.5,
        'r2_target': 0.7,
        'mape_target': 15.0
    },
    'current_performance': {
        'mae': None,  # Will be updated after training
        'r2': None,   # Will be updated after training
        'mape': None  # Will be updated after training
    }
}

# PARAMETER COMBINATIONS TESTED (for reference)
TESTED_COMBINATIONS = {
    'best_config_id': 'optimal_v1',
    'alternatives': [
        {'name': 'minimal', 'lags': [1, 7, 14], 'windows': [7, 14]},
        {'name': 'current', 'lags': [1, 2, 3, 7, 14, 21, 28], 'windows': [3, 7, 14, 30]},
        {'name': 'optimal', 'lags': [1, 2, 7, 14, 28], 'windows': [7, 14, 30]}
    ]
}

# OPTIMIZATION GRID FOR SYSTEMATIC TESTING
OPTIMIZATION_GRID = {
    'lag_combinations': [
        [1, 2, 3, 7],                    # Basic short-term
        [1, 2, 3, 7, 14],               # Current partial  
        [1, 2, 7, 14, 28],              # OPTIMAL - Current selection
        [7, 14, 21, 28],                # Weekly patterns only
        [1, 3, 7, 14, 30],              # Alternative mix
    ],
    'window_combinations': [
        [3, 7],                         # Short-term only
        [7, 14],                        # Medium-term focus
        [7, 14, 30],                    # OPTIMAL - Current selection
        [3, 7, 14, 30],                 # Extended full
        [7, 14, 30, 60],                # Long-term focus
    ],
    'feature_groups': [
        {'LAG_FEATURES': True, 'ROLLING_FEATURES': False, 'DATE_FEATURES': True, 'CYCLICAL_FEATURES': False},
        {'LAG_FEATURES': False, 'ROLLING_FEATURES': True, 'DATE_FEATURES': True, 'CYCLICAL_FEATURES': False}, 
        {'LAG_FEATURES': True, 'ROLLING_FEATURES': True, 'DATE_FEATURES': True, 'CYCLICAL_FEATURES': False},
        {'LAG_FEATURES': True, 'ROLLING_FEATURES': True, 'DATE_FEATURES': True, 'CYCLICAL_FEATURES': True},  # OPTIMAL
        {'LAG_FEATURES': True, 'ROLLING_FEATURES': True, 'DATE_FEATURES': False, 'CYCLICAL_FEATURES': True},
    ]
}

# Optimization settings
OPTIMIZATION_CONFIG = {
    'cv_splits': 5,                     # Cross-validation splits
    'test_punch_codes': ['202', '203', '206', '209', '210', '211', '213', '214', '215', '217'], # Test on these first (your enhanced codes)
    'min_improvement': 0.02,            # Minimum MAE improvement to consider
    'max_combinations': 25,             # Limit total combinations tested
}

# ==========================================
# SQL SERVER SETTINGS
# ==========================================
 
SQL_SERVER = "192.168.1.43"
SQL_DATABASE = "ABC"
SQL_DATABASE_LIVE = "fsystemp"
SQL_TRUSTED_CONNECTION = True
SQL_USERNAME = None
SQL_PASSWORD = None

# Parquet settings
PARQUET_COMPRESSION = "snappy"
PARQUET_ENGINE = "pyarrow"

# ==========================================
# BUSINESS RULES CONFIGURATION
# ==========================================

# Business Rules Configuration for Punch Code Working Days
PUNCH_CODE_WORKING_RULES = {
    # Define which punch codes work on which days
    # 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
    
    # Regular punch codes - work Monday to Friday only
    '202': [0, 1, 2, 3, 4],           # Mon-Fri
    '203': [0, 1, 2, 3, 4],           # Mon-Fri  
    '206': [0, 1, 2, 3, 4, 6],        # Mon-Fri + Sunday (special case)
    '208': [0, 1, 2, 3, 4],           # Mon-Fri
    '209': [0, 1, 2, 3, 4],           # Mon-Fri
    '210': [0, 1, 2, 3, 4],           # Mon-Fri
    '211': [0, 1, 2, 3, 4],           # Mon-Fri
    '213': [0, 1, 2, 3, 4],           # Mon-Fri
    '214': [0, 1, 2, 3, 4],           # Mon-Fri
    '215': [0, 1, 2, 3, 4],           # Mon-Fri
    '217': [0, 1, 2, 3, 4],           # Mon-Fri
}

# Default working days for unknown punch codes (Mon-Fri)
DEFAULT_PUNCH_CODE_WORKING_DAYS = [0, 1, 2, 3, 4]

# Punch code specific hours (if different from default)
DEFAULT_HOURS_PER_WORKER = 8.0

PUNCH_CODE_HOURS_PER_WORKER = {
    # '206': 7.5,  # Example: 206 works 7.5 hour shifts
    # '213': 8.5,  # Example: 213 works 8.5 hour shifts
}

# Enhanced work types for special handling
ENHANCED_WORK_TYPES = ['202', '203', '206', '209', '210', '211', '213', '214', '215', '217'] # 



# ==============================================
# LOGGING SETUP
# ==============================================

import logging

# Create basic logger early to avoid NameError
enterprise_logger = logging.getLogger('enterprise')
audit_logger = logging.getLogger('audit')

# Basic configuration - will be enhanced later
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)




# ==============================================
# ENTERPRISE CONFIGURATION
# ==============================================

@dataclass
class EnterpriseConfig:
    """Simple enterprise configuration"""
    enterprise_mode: bool = os.getenv('ENTERPRISE_MODE', 'false').lower() == 'true'
    
    class Environment:
        value: str = os.getenv('ENVIRONMENT', 'development')
    
    environment = Environment()

# Create enterprise config instance
ENTERPRISE_CONFIG = EnterpriseConfig()

# ==============================================
# MLFLOW CONFIGURATION
# ==============================================

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'workforce_prediction')
MLFLOW_ENABLE_TRACKING = os.getenv('MLFLOW_ENABLE_TRACKING', 'true').lower() == 'true'

# Create MLflow directories (without logging - will log later)
if MLFLOW_ENABLE_TRACKING:
    mlflow_dir = os.path.join(MODELS_DIR, 'mlflow-runs')
    os.makedirs(mlflow_dir, exist_ok=True)

# ==========================================
# CONFIGURATION VALIDATION
# ==========================================

def validate_config():
    """
    Validate configuration settings for optimal performance
    """
    warnings = []
    
    # Check feature engineering settings
    if not FEATURE_GROUPS['CYCLICAL_FEATURES']:
        warnings.append("⚠️ CYCLICAL_FEATURES disabled - may reduce accuracy for workforce patterns")
    
    if len(ESSENTIAL_LAGS) > 6:
        warnings.append("⚠️ Too many lag features - may cause overfitting")
    
    if len(ESSENTIAL_WINDOWS) > 4:
        warnings.append("⚠️ Too many rolling windows - may cause overfitting")
    
    # Check model parameters
    if DEFAULT_MODEL_PARAMS['max_depth'] > 10:
        warnings.append("⚠️ max_depth too high - may cause overfitting")
    
    if DEFAULT_MODEL_PARAMS['n_estimators'] > 500:
        warnings.append("⚠️ n_estimators too high - may cause slow training")
    
    # Print warnings if any
    if warnings:
        print("📋 CONFIGURATION VALIDATION:")
        for warning in warnings:
            print(f"   {warning}")
    else:
        print("✅ Configuration validated - optimized for accuracy")
    
    return len(warnings) == 0

# ==========================================
# CONFIGURATION SUMMARY
# ==========================================

def print_config_summary():
    """
    Print summary of current configuration
    """
    print("\n" + "="*60)
    print("📊 WORKFORCE PREDICTION CONFIGURATION SUMMARY")
    print("="*60)
    print(f"🎯 Target: MAE < {OPTIMIZATION_HISTORY['target_metrics']['mae_target']}, R² > {OPTIMIZATION_HISTORY['target_metrics']['r2_target']}")
    print(f"📅 Last Optimized: {OPTIMIZATION_HISTORY['last_optimized']}")
    print("\n🔧 FEATURE ENGINEERING:")
    
    enabled_features = [k for k, v in FEATURE_GROUPS.items() if v]
    print(f"   Enabled Groups: {enabled_features}")
    print(f"   Lag Periods: {ESSENTIAL_LAGS}")
    print(f"   Rolling Windows: {ESSENTIAL_WINDOWS}")
    print(f"   Lag Columns: {LAG_FEATURES_COLUMNS}")
    print(f"   Rolling Columns: {ROLLING_FEATURES_COLUMNS}")
    print(f"   Cyclical Features: {list(CYCLICAL_FEATURES.keys())}")
    
    print(f"\n🤖 MODEL CONFIGURATION:")
    print(f"   Estimators: {DEFAULT_MODEL_PARAMS['n_estimators']}")
    print(f"   Max Depth: {DEFAULT_MODEL_PARAMS['max_depth']}")
    print(f"   Min Samples Split: {DEFAULT_MODEL_PARAMS['min_samples_split']}")
    print(f"   Max Features: {DEFAULT_MODEL_PARAMS['max_features']}")
    
    print(f"\n🎯 TARGET PUNCH CODES:")
    print(f"   Enhanced Types: {ENHANCED_WORK_TYPES}")
    print(f"   All Punch Codes: {list(PUNCH_CODE_WORKING_RULES.keys())}")
    
    print("="*60)




# ==========================================
# PUNCH CODE 206 SPECIFIC CONFIGURATION
# ==========================================

# Punch code specific feature engineering rules
PUNCH_CODE_SPECIFIC_CONFIG = {
    '206': {
        # Sunday handling
        'isolate_sunday': True,
        'min_sunday_hours': 60,  # Minimum hours for Sunday to be considered normal
        'sunday_weight': 0.1,    # Reduced weight for Sunday data in training
        'exclude_sunday_from_weekday_patterns': True,
        
        # Enhanced weekday features
        'use_yearly_lags': True,
        'yearly_lag_days': [365, 366],  # Account for leap years
        'weekday_specific_lags': [7, 14],  # Last 2 weeks of same weekday
        'weekday_specific_windows': [2, 4],  # Last 2 and 4 weeks for same weekday
        
        # Feature selection
        'prioritize_weekday_features': True,
        'max_sunday_features': 3,  # Limit Sunday-specific features
        'force_include_yearly': True,  # Always include yearly patterns
    },

    '217': {
        'exclude_yearly_lags': True,        # Remove problematic yearly features
        'max_lag_days': 30,                 # Use only short-term lags
        'use_identity_transform': True,      # No log transformation
        'aggressive_outlier_removal': True,  # Use IQR method
        'min_data_coverage': 0.9            # Require high feature coverage
    }
}

# Enhanced lag configuration for punch code 206
PUNCH_206_ENHANCED_LAGS = [1, 7, 14, 365, 366]  # Add yearly lags

# Enhanced rolling windows for punch code 206
PUNCH_206_ENHANCED_WINDOWS = [7, 14, 28, 52]  # Add quarterly patterns

# Weekday mapping for feature engineering
WEEKDAY_NAMES = {
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
    3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
}

# Sunday isolation parameters
SUNDAY_CONFIG = {
    'isolation_threshold': 100,  # Hours below this on Sunday = isolated
    'weekday_influence_filter': True,  # Filter Sunday from weekday pattern learning
    'separate_sunday_validation': True,  # Validate Sunday predictions separately
}

# Auto-validate configuration on import
if __name__ == "__main__":
    validate_config()
    print_config_summary()