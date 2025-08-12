"""
Punch Code Specific Configurations for Time-Series Training
Handles working days, constraints, and optimization parameters per punch code
"""
from typing import Dict, List, Any
import pandas as pd

class PunchCodeConfig:
    """Configuration manager for punch-code specific settings"""
    
    # Sunday constraints for punch code 206
    SUNDAY_MAX_WORKERS_206 = 8
    
    # Working day patterns (leverages existing holiday_utils)
    WORKING_DAY_PATTERNS = {
        '206': {
            'works_weekends': True,
            'works_sundays': True,
            'sunday_max_workers': SUNDAY_MAX_WORKERS_206,
            'exclude_swedish_holidays': True
        },
        '217': {
            'works_weekends': False,
            'works_sundays': False,
            'exclude_swedish_holidays': True
        },
        'default': {
            'works_weekends': False,
            'works_sundays': False,
            'exclude_swedish_holidays': True
        }
    }
    
    # Training complexity levels (simple → complex)
    TRAINING_COMPLEXITY = {
        # Simple punch codes (good for testing approach)
        'simple': ['202', '203', '210', '214', '215'],
        
        # Medium complexity 
        'medium': ['217', '209', '211', '213'],
        
        # Complex (special rules, weekend patterns)
        'complex': ['206']
    }
    
    # Hyperparameters optimized per punch code
    HYPERPARAMETERS = {
        '217': {
            'learning_rate': 0.01,
            'min_child_samples': 20,
            'n_estimators': 2000,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        },
        '206': {
            'learning_rate': 0.015,
            'min_child_samples': 15,
            'n_estimators': 1500,
            'num_leaves': 63,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'default': {
            'learning_rate': 0.01,
            'min_child_samples': 25,
            'n_estimators': 1000,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    }
    
    # Feature engineering configuration per punch code
    FEATURE_CONFIGS = {
        '217': {
            'lags': [1, 7, 14, 21, 30, 365, 366],
            'rolling_windows': [7, 14, 28],
            'use_cyclical_features': True,
            'use_interaction_features': True,
            'use_year_over_year': True,
            'use_difference_features': True,
            'exclude_weekends': True,
            'data_start_date': '2019-07-01'
        },
        '206': {
            'lags': [1, 7, 14, 30, 365, 366],
            'rolling_windows': [7, 14, 28, 52],
            'use_cyclical_features': True,
            'use_interaction_features': True,
            'use_year_over_year': True,
            'use_difference_features': True,
            'sunday_special_handling': True,
            'saturday_patterns': True,
            'exclude_weekends': False,  # 206 works weekends
            'data_start_date': '2019-07-01'
        },
        'default': {
            'lags': [1, 7, 14, 30],
            'rolling_windows': [7, 14],
            'use_cyclical_features': True,
            'use_interaction_features': False,
            'use_year_over_year': False,
            'use_difference_features': False,
            'exclude_weekends': True,
            'data_start_date': '2020-01-01'
        }
    }
    
    # Outlier handling strategies per punch code
    OUTLIER_CONFIGS = {
        '217': {
            'method': 'iqr',
            'lower_factor': 1.5,
            'upper_factor': 1.5,
            'preserve_peaks': False
        },
        '214': {
            'method': 'iqr', 
            'lower_factor': 2.0,
            'upper_factor': 3.0,
            'preserve_peaks': True  # 214 has legitimate high values
        },
        '206': {
            'method': 'iqr',
            'lower_factor': 1.5,
            'upper_factor': 2.0,
            'sunday_special': True  # Different outlier rules for Sunday
        },
        'default': {
            'method': 'std',
            'std_factor': 4.0
        }
    }
    
    # Validation strategies per punch code
    VALIDATION_CONFIGS = {
        '217': {
            'method': 'time_series_split',
            'n_splits': 5,
            'test_size': 0.2
        },
        '206': {
            'method': 'time_series_split',
            'n_splits': 3,  # Less splits due to weekend complexity
            'test_size': 0.2
        },
        'default': {
            'method': 'time_series_split',
            'n_splits': 5,
            'test_size': 0.2
        }
    }
    
    @staticmethod
    def get_punch_code_config(punch_code: str) -> Dict[str, Any]:
        """Get complete configuration for a specific punch code"""
        
        config = {
            'working_days': PunchCodeConfig.WORKING_DAY_PATTERNS.get(
                punch_code, PunchCodeConfig.WORKING_DAY_PATTERNS['default']
            ),
            'hyperparameters': PunchCodeConfig.HYPERPARAMETERS.get(
                punch_code, PunchCodeConfig.HYPERPARAMETERS['default']
            ),
            'features': PunchCodeConfig.FEATURE_CONFIGS.get(
                punch_code, PunchCodeConfig.FEATURE_CONFIGS['default']
            ),
            'outliers': PunchCodeConfig.OUTLIER_CONFIGS.get(
                punch_code, PunchCodeConfig.OUTLIER_CONFIGS['default']
            ),
            'validation': PunchCodeConfig.VALIDATION_CONFIGS.get(
                punch_code, PunchCodeConfig.VALIDATION_CONFIGS['default']
            )
        }
        
        return config
    
    @staticmethod
    def get_complexity_level(punch_code: str) -> str:
        """Get complexity level for a punch code"""
        for level, codes in PunchCodeConfig.TRAINING_COMPLEXITY.items():
            if punch_code in codes:
                return level
        return 'medium'  # Default
    
    @staticmethod
    def get_recommended_training_order() -> List[str]:
        """Get recommended order for implementing punch codes"""
        order = []
        
        # Start with proven code
        order.extend(['217'])  # Already working
        
        # Then simple codes
        order.extend(PunchCodeConfig.TRAINING_COMPLEXITY['simple'])
        
        # Then medium complexity
        order.extend([code for code in PunchCodeConfig.TRAINING_COMPLEXITY['medium'] if code != '217'])
        
        # Finally complex codes
        order.extend(PunchCodeConfig.TRAINING_COMPLEXITY['complex'])
        
        return order
    
    @staticmethod
    def print_punch_code_info(punch_code: str):
        """Print detailed information about a punch code configuration"""
        
        config = PunchCodeConfig.get_punch_code_config(punch_code)
        complexity = PunchCodeConfig.get_complexity_level(punch_code)
        
        print(f"\n📋 PUNCH CODE {punch_code} CONFIGURATION:")
        print(f"   Complexity Level: {complexity.upper()}")
        print(f"   Works Weekends: {config['working_days']['works_weekends']}")
        print(f"   Works Sundays: {config['working_days']['works_sundays']}")
        
        if punch_code == '206':
            print(f"   Sunday Max Workers: {config['working_days']['sunday_max_workers']}")
        
        print(f"   Feature Lags: {config['features']['lags']}")
        print(f"   Rolling Windows: {config['features']['rolling_windows']}")
        print(f"   Cyclical Features: {config['features']['use_cyclical_features']}")
        print(f"   Interaction Features: {config['features']['use_interaction_features']}")
        print(f"   Year-over-Year: {config['features']['use_year_over_year']}")
        
        print(f"   Hyperparameters:")
        for key, value in config['hyperparameters'].items():
            print(f"     {key}: {value}")


# Performance tracking
class TrainingTracker:
    """Track training performance across punch codes"""
    
    def __init__(self):
        self.results = {}
    
    def record_result(self, punch_code: str, metrics: Dict):
        """Record training results for a punch code"""
        self.results[punch_code] = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'status': 'success' if metrics['cv_r2_mean'] > 0.7 else 'needs_improvement'
        }
    
    def print_summary(self):
        """Print training summary"""
        print("\n📊 TRAINING RESULTS SUMMARY:")
        print("=" * 50)
        
        for punch_code, result in self.results.items():
            status_emoji = "✅" if result['status'] == 'success' else "⚠️"
            metrics = result['metrics']
            
            print(f"{status_emoji} Punch Code {punch_code}:")
            print(f"   R² = {metrics['cv_r2_mean']:.4f}")
            print(f"   MAE = {metrics['cv_mae_mean']:.4f}")
            print(f"   Status: {result['status']}")


if __name__ == "__main__":
    main()