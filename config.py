"""
Configuration settings for the Work Utilization Prediction application.
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

# DEFAULT_MODEL_PARAMS = {
#     "n_estimators": 500,  # More trees for better accuracy
#     "max_depth": 15,      # Deeper trees for complex patterns
#     "min_samples_split": 3,  # More sensitive to patterns
#     "min_samples_leaf": 1,   # Allow finer granularity
#     "random_state": 42,
#     "max_features": 0.8,     # Use more features
#     "bootstrap": True,
#     "oob_score": True,       # Out-of-bag scoring
# }

DEFAULT_MODEL_PARAMS = {
    "n_estimators": 300,      # ✅ Fewer trees to prevent memorization
    "max_depth": 6,           # ✅ Shallow trees for generalization
    "min_samples_split": 10,  # ✅ Conservative splitting
    "min_samples_leaf": 5,    # ✅ Larger leaves for stability
    "max_features": "sqrt",   # ✅ Feature subsampling
    "bootstrap": True,
    "random_state": 42,
}

# Feature engineering settingsa
# LAG_DAYS = [1, 2, 3, 7, 14, 30]  # Default lag days for feature engineering
# ROLLING_WINDOWS = [7, 14, 30, 90]    # Default rolling windows for feature engineering

LAG_DAYS = [1, 2, 7, 28, 365]  # 28 for true monthly cycle
ROLLING_WINDOWS = [7, 21, 30, 90]  # 21 for 3-week patterns
 
# SQL Server settings
SQL_SERVER = "192.168.1.43"
SQL_DATABASE = "ABC"
SQL_DATABASE_LIVE = "fsystemp"
SQL_TRUSTED_CONNECTION = True
SQL_USERNAME = None
SQL_PASSWORD = None

# Parquet settings
PARQUET_COMPRESSION = "snappy"
PARQUET_ENGINE = "pyarrow"

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


# ==========================================
# FEATURE SELECTION CONFIGURATION
# ==========================================

# Core feature categories - Enable/Disable groups to prevent overfitting
FEATURE_GROUPS = {
    'LAG_FEATURES': True,           # Essential: NoOfMan_lag_1, lag_7, etc.
    'ROLLING_FEATURES': False,       # Essential: rolling_mean_7, etc.
    'DATE_FEATURES': True,          # Essential: DayOfWeek, Month, etc.
    # 'PRODUCTIVITY_FEATURES': False,  # New: Workers_per_Hour, etc.
    'CYCLICAL_FEATURES': False,     # Optional: Sin/Cos transforms ENABLE for better patterns
    'TREND_FEATURES': False,        # Optional: Trend calculations ENABLE for momentum
    'PATTERN_FEATURES': False,      # Optional: Same day patterns ENABLE for seasonality
}


# Productivity features to create (only if PRODUCTIVITY_FEATURES=True)
PRODUCTIVITY_FEATURES = [
    'Workers_per_Hour',
    'Quantity_per_Hour', 
    'Workload_Density',
    'KPI_Performance'
]

# # Essential lag features (reduce from current LAG_DAYS to prevent overfitting)
# ESSENTIAL_LAGS = [1, 7, 28]  # Only most important: yesterday, last week, last month

# # Essential rolling windows (reduce from current ROLLING_WINDOWS)
# ESSENTIAL_WINDOWS = [7, 30]  # Only weekly and monthly averages

ESSENTIAL_LAGS = [1, 2, 3, 7, 14, 21, 28]  # More granular lags
ESSENTIAL_WINDOWS = [3, 7, 14, 30]     

LAG_FEATURES_COLUMNS = ['Hours', 'Quantity', 'SystemHours']
ROLLING_FEATURES_COLUMNS = ['Hours', 'Quantity', 'SystemHours']
CYCLICAL_FEATURES = {'DayOfWeek': 7, 'Month': 12}

# Date features to include
DATE_FEATURES = {
    'categorical': ['DayOfWeek_feat', 'Month_feat'],
    'numeric': ['IsWeekend_feat']
}


# ==============================================
# BASIC LOGGING SETUP (SIMPLIFIED)
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
# ENTERPRISE CONFIGURATION (Simple)
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
# MLFLOW CONFIGURATION (Simple)
# ==============================================

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'workforce_prediction')
MLFLOW_ENABLE_TRACKING = os.getenv('MLFLOW_ENABLE_TRACKING', 'true').lower() == 'true'

# Create MLflow directories (without logging - will log later)
if MLFLOW_ENABLE_TRACKING:
    mlflow_dir = os.path.join(MODELS_DIR, 'mlflow-runs')
    os.makedirs(mlflow_dir, exist_ok=True)
