"""
Configuration settings for the Work Utilization Prediction application.
"""
import os
from pathlib import Path

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

# Model settings
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42
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
# TIERED FEATURE CONFIGURATION SYSTEM
# ==========================================

# Feature tier controls (can be turned on/off)
FEATURE_TIERS = {
    'BASIC': True,        # Essential features - always recommended
    'INTERMEDIATE': True, # Enhanced features for better accuracy  
    'ADVANCED': True      # Sophisticated features for maximum performance
}

# ==========================================
# BASIC FEATURES (Tier 1) - Essential
# ==========================================
BASIC_FEATURES = {
    # Core date features
    'DATE_FEATURES': {
        'categorical': ['DayOfWeek_feat', 'Month_feat'],
        'numeric': ['IsWeekend_feat']
    },
    
    # Essential lag features - most recent patterns
    'LAG_FEATURES': {
        'NoOfMan': [1, 7],  # Yesterday and same day last week
    },
    
    # Basic trend indicators
    'ROLLING_FEATURES': {
        'NoOfMan': {
            'windows': [7],  # Weekly averages
            'functions': ['mean']
        }
    }
}

# ==========================================
# INTERMEDIATE FEATURES (Tier 2) - Enhanced
# ==========================================
INTERMEDIATE_FEATURES = {
    # Extended date features
    'DATE_FEATURES': {
        'categorical': ['Quarter'],
        'numeric': ['Year_feat', 'DayOfMonth', 'WeekOfYear']
    },
    
    # More comprehensive lag patterns
    'LAG_FEATURES': {
        'NoOfMan': [2, 3, 14, 30],  # Short to medium term patterns
    },
    
    # Enhanced rolling statistics
    'ROLLING_FEATURES': {
        'NoOfMan': {
            'windows': [14, 30],  # Bi-weekly and monthly patterns
            'functions': ['mean'],
            'extended_stats': {  # Additional stats for first window only
                'window': 7,
                'functions': ['max', 'min', 'std']
            }
        }
    },
    
    # Pattern recognition features
    'PATTERN_FEATURES': [
        'NoOfMan_same_dow_lag',    # Same day of week pattern
        'NoOfMan_same_dom_lag',    # Same day of month pattern
    ],
    
    # Trend features
    'TREND_FEATURES': [
        ('NoOfMan_7day_trend', 'NoOfMan', 1, 7),   # Week-over-week trend
        ('NoOfMan_1day_trend', 'NoOfMan', 1, 2),   # Day-over-day trend
    ]
}

# ==========================================
# ADVANCED FEATURES (Tier 3) - Sophisticated
# ==========================================
ADVANCED_FEATURES = {
    # Extended date intelligence
    'DATE_FEATURES': {
        'categorical': [],
        'numeric': ['DayOfYear']
    },
    
    # Long-term patterns
    'LAG_FEATURES': {
        'NoOfMan': [90, 365],  # Seasonal and yearly patterns
    },
    
    # Extended rolling windows
    'ROLLING_FEATURES': {
        'NoOfMan': {
            'windows': [90],  # Quarterly patterns
            'functions': ['mean']
        }
    },
    
    # Productivity metrics (if data available)
    'PRODUCTIVITY_FEATURES': {
        'LAG_FEATURES': {
            'Quantity': [1, 7, 30],
            'Hours': [1, 7],
            'SystemHours': [1, 7],
            'ResourceKPI': [1, 7],
            'SystemKPI': [1, 7]
        },
        'ROLLING_FEATURES': {
            'Quantity': {
                'windows': [7, 30],
                'functions': ['mean']
            },
            'Hours': {
                'windows': [7],
                'functions': ['mean']
            }
        },
        'DERIVED_FEATURES': [
            'Hours_SystemHours_Ratio',
            'Quantity_per_Worker', 
            'Hours_per_Quantity',
            'SystemHours_per_Quantity',
            'Combined_KPI',
            'Workers_Predicted_from_Quantity'
        ],
        'PATTERN_FEATURES': [
            'Quantity_same_dow_lag'
        ],
        'TREND_FEATURES': [
            ('Quantity_7day_trend', 'Quantity', 1, 7),
            ('Hours_7day_trend', 'Hours', 1, 7)
        ]
    },
    
    # Business logic features
    'BUSINESS_FEATURES': [
        'Workload_Intensity',      # High/Medium/Low workload indicator
        'Seasonal_Factor',         # Seasonal adjustment factor
        'Capacity_Utilization',    # How close to maximum capacity
    ]
}

# ==========================================
# FEATURE AVAILABILITY DETECTION
# ==========================================
FEATURE_AVAILABILITY = {
    # Data columns required for productivity features
    'PRODUCTIVITY_REQUIRED_COLUMNS': ['Quantity', 'Hours', 'SystemHours', 'ResourceKPI', 'SystemKPI'],
    
    # Minimum data requirements
    'MIN_DATA_REQUIREMENTS': {
        'BASIC': 14,        # Need at least 2 weeks of data
        'INTERMEDIATE': 45, # Need at least 6 weeks of data  
        'ADVANCED': 180     # Need at least 6 months of data
    }
}