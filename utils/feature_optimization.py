"""
Feature Engineering Parameter Optimization System
Systematic approach to find optimal feature combinations for maximum accuracy
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import itertools
from datetime import datetime
import config
from utils.feature_engineering import EnhancedFeatureTransformer

logger = logging.getLogger(__name__)

class FeatureParameterOptimizer:
    """
    Systematic optimization of feature engineering parameters
    Uses cross-validation to prevent overfitting and ensure robust results
    """
    
    def __init__(self, df, target_punch_codes=None):
        """
        Initialize optimizer with data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with Date, WorkType, Hours columns
        target_punch_codes : list
            Punch codes to optimize for (default: enhanced work types from config)
        """
        self.df = df.copy()
        self.target_punch_codes = target_punch_codes or config.ENHANCED_WORK_TYPES
        self.results = []
        self.best_config = None
        self.baseline_performance = {}
        
        # Ensure data is properly formatted
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(['WorkType', 'Date'])
        
        logger.info(f"🎯 Initialized optimizer for punch codes: {self.target_punch_codes}")
        logger.info(f"📊 Data shape: {self.df.shape}, Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")

    def create_parameter_combinations(self):
        """
        Generate systematic parameter combinations for testing
        
        Returns:
        --------
        list
            List of parameter dictionaries to test
        """
        combinations = []
        
        # Get optimization grid from config
        lag_combos = config.OPTIMIZATION_GRID['lag_combinations']
        window_combos = config.OPTIMIZATION_GRID['window_combinations']  
        feature_groups = config.OPTIMIZATION_GRID['feature_groups']
        
        # Create combinations
        for feature_group in feature_groups:
            for lags in lag_combos:
                for windows in window_combos:
                    # Skip invalid combinations
                    if feature_group['LAG_FEATURES'] and not lags:
                        continue
                    if feature_group['ROLLING_FEATURES'] and not windows:
                        continue
                        
                    combination = {
                        'feature_groups': feature_group,
                        'lag_days': lags if feature_group['LAG_FEATURES'] else [],
                        'rolling_windows': windows if feature_group['ROLLING_FEATURES'] else [],
                        'lag_columns': ['Hours', 'Quantity', 'SystemHours'],
                        'rolling_columns': ['Hours', 'Quantity', 'SystemHours'],
                    }
                    combinations.append(combination)
        
        # Limit combinations to prevent excessive runtime
        max_combinations = config.OPTIMIZATION_CONFIG['max_combinations']
        if len(combinations) > max_combinations:
            logger.info(f"⚠️ Limiting to {max_combinations} combinations (from {len(combinations)} total)")
            combinations = combinations[:max_combinations]
            
        logger.info(f"🔍 Generated {len(combinations)} parameter combinations to test")
        return combinations

    def apply_feature_config(self, param_config):
        """
        Temporarily apply parameter configuration to config module
        
        Parameters:
        -----------
        param_config : dict
            Parameter configuration to apply
        """
        # Store original config
        original_feature_groups = config.FEATURE_GROUPS.copy()
        original_lags = config.ESSENTIAL_LAGS.copy()
        original_windows = config.ESSENTIAL_WINDOWS.copy()
        original_lag_cols = config.LAG_FEATURES_COLUMNS.copy()
        original_rolling_cols = config.ROLLING_FEATURES_COLUMNS.copy()
        
        # Apply new config
        config.FEATURE_GROUPS.update(param_config['feature_groups'])
        config.ESSENTIAL_LAGS = param_config['lag_days']
        config.ESSENTIAL_WINDOWS = param_config['rolling_windows']
        config.LAG_FEATURES_COLUMNS = param_config['lag_columns']
        config.ROLLING_FEATURES_COLUMNS = param_config['rolling_columns']
        
        return {
            'FEATURE_GROUPS': original_feature_groups,
            'ESSENTIAL_LAGS': original_lags,
            'ESSENTIAL_WINDOWS': original_windows,
            'LAG_FEATURES_COLUMNS': original_lag_cols,
            'ROLLING_FEATURES_COLUMNS': original_rolling_cols
        }

    def restore_feature_config(self, original_config):
        """
        Restore original configuration
        
        Parameters:
        -----------
        original_config : dict
            Original configuration to restore
        """
        config.FEATURE_GROUPS = original_config['FEATURE_GROUPS']
        config.ESSENTIAL_LAGS = original_config['ESSENTIAL_LAGS']
        config.ESSENTIAL_WINDOWS = original_config['ESSENTIAL_WINDOWS']
        config.LAG_FEATURES_COLUMNS = original_config['LAG_FEATURES_COLUMNS']
        config.ROLLING_FEATURES_COLUMNS = original_config['ROLLING_FEATURES_COLUMNS']

    def evaluate_parameter_combination(self, param_config, punch_code):
        """
        Evaluate a single parameter combination using cross-validation
        
        Parameters:
        -----------
        param_config : dict
            Parameter configuration to test
        punch_code : str
            Punch code to evaluate
            
        Returns:
        --------
        dict
            Performance metrics
        """
        try:
            # Filter data for this punch code
            punch_data = self.df[self.df['WorkType'] == punch_code].copy()
            
            if len(punch_data) < 50:  # Need minimum data
                logger.warning(f"⚠️ Insufficient data for {punch_code}: {len(punch_data)} records")
                return None
                
            # Apply parameter configuration
            original_config = self.apply_feature_config(param_config)
            
            try:
                # Create features with new configuration
                transformer = EnhancedFeatureTransformer()
                transformed_data = transformer.fit_transform(punch_data)
                
                # Prepare features and target
                feature_cols = [col for col in transformed_data.columns 
                              if col not in ['Date', 'WorkType', 'Hours', 'Quantity', 'SystemHours']]
                
                if len(feature_cols) == 0:
                    logger.warning(f"⚠️ No features generated for {punch_code}")
                    return None
                    
                X = transformed_data[feature_cols].fillna(0)
                y = transformed_data['Hours']
                
                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=config.OPTIMIZATION_CONFIG['cv_splits'])
                cv_scores = []
                cv_r2_scores = []
                
                # Model pipeline
                model = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('rf', RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        random_state=42,
                        n_jobs=-1
                    ))
                ])
                
                # Perform cross-validation
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    cv_scores.append(mae)
                    cv_r2_scores.append(r2)
                
                # Calculate metrics
                avg_mae = np.mean(cv_scores)
                std_mae = np.std(cv_scores)
                avg_r2 = np.mean(cv_r2_scores)
                
                return {
                    'punch_code': punch_code,
                    'avg_mae': avg_mae,
                    'std_mae': std_mae,
                    'avg_r2': avg_r2,
                    'feature_count': len(feature_cols),
                    'data_points': len(punch_data),
                    'cv_scores': cv_scores
                }
                
            finally:
                # Always restore original configuration
                self.restore_feature_config(original_config)
                
        except Exception as e:
            logger.error(f"❌ Error evaluating {punch_code}: {str(e)}")
            return None

    def run_optimization(self):
        """
        Run complete parameter optimization
        
        Returns:
        --------
        dict
            Best configuration and results
        """
        logger.info("🚀 Starting systematic parameter optimization...")
        
        # Generate parameter combinations
        combinations = self.create_parameter_combinations()
        
        # Store all results
        all_results = []
        
        # Test each combination
        for i, param_config in enumerate(combinations):
            logger.info(f"🔍 Testing combination {i+1}/{len(combinations)}")
            
            combination_results = {
                'config_id': i,
                'param_config': param_config,
                'punch_results': {},
                'avg_performance': {}
            }
            
            # Test on each punch code
            punch_maes = []
            punch_r2s = []
            
            for punch_code in self.target_punch_codes:
                result = self.evaluate_parameter_combination(param_config, punch_code)
                
                if result:
                    combination_results['punch_results'][punch_code] = result
                    punch_maes.append(result['avg_mae'])
                    punch_r2s.append(result['avg_r2'])
            
            # Calculate average performance across punch codes
            if punch_maes:
                combination_results['avg_performance'] = {
                    'avg_mae': np.mean(punch_maes),
                    'avg_r2': np.mean(punch_r2s),
                    'valid_punch_codes': len(punch_maes)
                }
                
                all_results.append(combination_results)
                
                logger.info(f"  Result: MAE={np.mean(punch_maes):.3f}, R²={np.mean(punch_r2s):.3f}")
        
        # Find best configuration
        if all_results:
            best_result = min(all_results, key=lambda x: x['avg_performance']['avg_mae'])
            
            logger.info("🎯 OPTIMIZATION COMPLETE!")
            logger.info(f"✅ Best MAE: {best_result['avg_performance']['avg_mae']:.3f}")
            logger.info(f"✅ Best R²: {best_result['avg_performance']['avg_r2']:.3f}")
            
            return {
                'best_config': best_result['param_config'],
                'best_performance': best_result['avg_performance'],
                'all_results': all_results,
                'summary': self.create_optimization_summary(all_results)
            }
        else:
            logger.error("❌ No valid results found")
            return None

    def create_optimization_summary(self, all_results):
        """
        Create summary of optimization results
        
        Parameters:
        -----------
        all_results : list
            All optimization results
            
        Returns:
        --------
        dict
            Summary statistics
        """
        if not all_results:
            return {}
            
        maes = [r['avg_performance']['avg_mae'] for r in all_results]
        r2s = [r['avg_performance']['avg_r2'] for r in all_results]
        
        return {
            'total_combinations_tested': len(all_results),
            'mae_range': {'min': min(maes), 'max': max(maes), 'std': np.std(maes)},
            'r2_range': {'min': min(r2s), 'max': max(r2s), 'std': np.std(r2s)},
            'improvement_potential': max(maes) - min(maes),
            'best_rank_percentile': (1 / len(all_results)) * 100
        }

def run_feature_optimization(df, target_punch_codes=None):
    """
    Convenience function to run complete optimization
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training data
    target_punch_codes : list
        Punch codes to optimize for
        
    Returns:
    --------
    dict
        Optimization results
    """
    optimizer = FeatureParameterOptimizer(df, target_punch_codes)
    return optimizer.run_optimization()