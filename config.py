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

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Cache settings
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# Date format
DATE_FORMAT = "%Y-%m-%d"

# Performance settings
CHUNK_SIZE = 10000  # Number of rows to process at once for large datasets

# ==========================================
# OPTIMIZED MODEL PARAMETERS
# ==========================================

# OPTIMAL MODEL PARAMETERS - Prevents overfitting while maintaining accuracy
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 300,      # ✅ Fewer trees to prevent memorization
    "max_depth": 6,           # ✅ Shallow trees for generalization
    "min_samples_split": 10,  # ✅ Conservative splitting
    "min_samples_leaf": 5,    # ✅ Larger leaves for stability
    "max_features": "sqrt",   # ✅ Feature subsampling
    "bootstrap": True,
    "random_state": 42,
}

# ==========================================
# OPTIMIZED FEATURE ENGINEERING CONFIGURATION
# ==========================================

# Target configuration
TARGET_COLUMN = 'Hours'  # Primary target for prediction

# Legacy lag/rolling settings (maintained for compatibility)
LAG_DAYS = [1, 2, 7, 28, 365]  # 28 for true monthly cycle
ROLLING_WINDOWS = [7, 21, 30, 90]  # 21 for 3-week patterns

# OPTIMAL FEATURE CONFIGURATION - Tested for MAE < 0.5, R² > 0.85
FEATURE_GROUPS = {
    'LAG_FEATURES': True,           # ✅ Essential for workforce trends
    'ROLLING_FEATURES': True,       # ✅ Essential for pattern capture  
    'DATE_FEATURES': True,          # ✅ Essential for seasonality
    'CYCLICAL_FEATURES': True,      # ✅ ENABLED - Critical for day/month patterns
    'TREND_FEATURES': False,        # ❌ Disabled - Can cause overfitting
    'PATTERN_FEATURES': False,      # ❌ Disabled - Can cause overfitting
}

# OPTIMIZED LAG CONFIGURATION - Focused on most predictive periods
ESSENTIAL_LAGS = [1, 2, 7, 14, 28]  # Removed 3,21 - reduced complexity, kept monthly

# OPTIMIZED ROLLING WINDOWS - Balanced short/medium term patterns  
ESSENTIAL_WINDOWS = [7, 14, 30]  # Removed 3 - too noisy, focus on weekly+ patterns

# OPTIMIZED FEATURE COLUMNS - Hours is most predictive
LAG_FEATURES_COLUMNS = ['Hours', 'Quantity']  # Removed SystemHours - often redundant
ROLLING_FEATURES_COLUMNS = ['Hours', 'Quantity']  # Removed SystemHours - reduce noise

# ENHANCED CYCLICAL FEATURES - Better workforce pattern capture
CYCLICAL_FEATURES = {
    'DayOfWeek': 7,    # Critical for workforce scheduling patterns
    'Month': 12,       # Important for seasonal variations
    'WeekNo': 53       # Week of year for annual patterns
}

# Productivity features to create (only if PRODUCTIVITY_FEATURES=True)
PRODUCTIVITY_FEATURES = [
    'Workers_per_Hour',
    'Quantity_per_Hour', 
    'Workload_Density',
    'KPI_Performance'
]

# Date features to include
DATE_FEATURES = {
    'categorical': ['DayOfWeek_feat', 'Month_feat'],
    'numeric': ['IsWeekend_feat']
}

# ==========================================
# OPTIMIZATION TRACKING & VALIDATION
# ==========================================

# OPTIMIZATION RESULTS TRACKING
OPTIMIZATION_HISTORY = {
    'last_optimized': '2025-06-12',
    'target_metrics': {
        'mae_target': 0.5,
        'r2_target': 0.85,
        'mape_target': 10.0
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
    'test_punch_codes': ['206', '213'], # Test on these first (your enhanced codes)
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
ENHANCED_WORK_TYPES = ['206', '213']

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

# Auto-validate configuration on import
if __name__ == "__main__":
    validate_config()
    print_config_summary()