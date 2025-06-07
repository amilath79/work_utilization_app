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


# ==============================================
# ENTERPRISE CONFIGURATION CLASSES
# ==============================================

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class MLflowConfig:
    """Enterprise MLflow Configuration"""
    tracking_uri: str
    artifact_root: str
    experiment_name: str
    enable_tracking: bool
    auto_log: bool
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'MLflowConfig':
        """Create MLflow config from environment variables"""
        return cls(
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI', f'file:///{MODELS_DIR}/mlflow-runs'),
            artifact_root=os.getenv('MLFLOW_ARTIFACT_ROOT', f'{MODELS_DIR}/mlflow-artifacts'),
            experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME', 'workforce_prediction'),
            enable_tracking=os.getenv('MLFLOW_ENABLE_TRACKING', 'true').lower() == 'true',
            auto_log=os.getenv('MLFLOW_AUTO_LOG', 'true').lower() == 'true',
            username=os.getenv('MLFLOW_TRACKING_USERNAME'),
            password=os.getenv('MLFLOW_TRACKING_PASSWORD'),
            token=os.getenv('MLFLOW_TRACKING_TOKEN')
        )
    
    def validate(self) -> None:
        """Validate MLflow configuration"""
        if self.enable_tracking:
            if not self.tracking_uri:
                raise ValueError("MLflow tracking URI is required when tracking is enabled")
            
            # Create directories if using file-based tracking
            if self.tracking_uri.startswith('file://'):
                tracking_path = self.tracking_uri.replace('file:///', '').replace('file://', '')
                os.makedirs(tracking_path, exist_ok=True)
                
            if self.artifact_root:
                os.makedirs(self.artifact_root, exist_ok=True)

@dataclass
class EnterpriseConfig:
    """Enterprise application configuration"""
    environment: Environment
    log_level: LogLevel
    enterprise_mode: bool
    audit_logging: bool
    security_headers: bool
    
    @classmethod
    def from_env(cls) -> 'EnterpriseConfig':
        """Create enterprise config from environment variables"""
        return cls(
            environment=Environment(os.getenv('ENVIRONMENT', 'development')),
            log_level=LogLevel(os.getenv('LOG_LEVEL', 'INFO')),
            enterprise_mode=os.getenv('ENTERPRISE_MODE', 'false').lower() == 'true',
            audit_logging=os.getenv('AUDIT_LOGGING', 'false').lower() == 'true',
            security_headers=os.getenv('SECURITY_HEADERS', 'true').lower() == 'true'
        )

# Initialize enterprise configurations
MLFLOW_CONFIG = MLflowConfig.from_env()
ENTERPRISE_CONFIG = EnterpriseConfig.from_env()

# Validate configurations
try:
    MLFLOW_CONFIG.validate()
except Exception as e:
    print(f"MLflow configuration error: {e}")
    MLFLOW_CONFIG.enable_tracking = False



# ==============================================
# ENTERPRISE LOGGING CONFIGURATION
# ==============================================

# Enterprise logging directories
ENTERPRISE_LOGS_DIR = os.path.join(BASE_DIR, "logs", "enterprise")
AUDIT_LOGS_DIR = os.path.join(BASE_DIR, "logs", "audit")
MODEL_LOGS_DIR = os.path.join(BASE_DIR, "logs", "models")

# Create logging directories
for log_dir in [ENTERPRISE_LOGS_DIR, AUDIT_LOGS_DIR, MODEL_LOGS_DIR]:
    os.makedirs(log_dir, exist_ok=True)

# Enterprise logging configuration
ENTERPRISE_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'enterprise': {
            'format': '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'audit': {
            'format': '%(asctime)s | AUDIT | %(levelname)s | %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'enterprise_file': {
            'level': ENTERPRISE_CONFIG.log_level.value,
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(ENTERPRISE_LOGS_DIR, 'application.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'enterprise'
        },
        'audit_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(AUDIT_LOGS_DIR, 'audit.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
            'formatter': 'audit'
        },
        'console': {
            'level': ENTERPRISE_CONFIG.log_level.value,
            'class': 'logging.StreamHandler',
            'formatter': 'enterprise'
        }
    },
    'loggers': {
        'enterprise': {
            'handlers': ['enterprise_file', 'console'],
            'level': ENTERPRISE_CONFIG.log_level.value,
            'propagate': False
        },
        'audit': {
            'handlers': ['audit_file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Apply logging configuration
import logging.config
logging.config.dictConfig(ENTERPRISE_LOGGING_CONFIG)

# Create enterprise loggers
enterprise_logger = logging.getLogger('enterprise')
audit_logger = logging.getLogger('audit')

#--------------------------------------------------------------

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

DEFAULT_MODEL_PARAMS = {
    "n_estimators": 500,  # More trees for better accuracy
    "max_depth": 15,      # Deeper trees for complex patterns
    "min_samples_split": 3,  # More sensitive to patterns
    "min_samples_leaf": 1,   # Allow finer granularity
    "random_state": 42,
    "max_features": 0.8,     # Use more features
    "bootstrap": True,
    "oob_score": True,       # Out-of-bag scoring
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
    'ROLLING_FEATURES': True,       # Essential: rolling_mean_7, etc.
    'DATE_FEATURES': True,          # Essential: DayOfWeek, Month, etc.
    'PRODUCTIVITY_FEATURES': True,  # New: Workers_per_Hour, etc.
    'CYCLICAL_FEATURES': True,     # Optional: Sin/Cos transforms ENABLE for better patterns
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

# Essential lag features (reduce from current LAG_DAYS to prevent overfitting)
ESSENTIAL_LAGS = [1, 7, 28]  # Only most important: yesterday, last week, last month

# Essential rolling windows (reduce from current ROLLING_WINDOWS)
ESSENTIAL_WINDOWS = [7, 30]  # Only weekly and monthly averages

# Date features to include
DATE_FEATURES = {
    'categorical': ['DayOfWeek_feat', 'Month_feat'],
    'numeric': ['IsWeekend_feat']
}


