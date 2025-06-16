import pandas as pd
import numpy as np
import logging
import json
import time
from datetime import datetime
from itertools import combinations, product
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

from utils.feature_engineering import (
    add_lag_features_by_group,
    add_rolling_features_by_group, 
    add_date_features,
    add_cyclical_features,
    add_trend_features,
    add_pattern_features
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_selection_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureCombinationSelector:
    
    def __init__(self, data_path: str, target_punch_codes: List[str]):
        self.data_path = data_path
        self.target_punch_codes = target_punch_codes
        self.results = []
        self.best_config = None
        self.best_score = float('inf')
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data following train_models2.py pattern"""
        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['punch_code', 'date'])
        
        # Filter target punch codes
        df = df[df['punch_code'].isin(self.target_punch_codes)]
        
        logger.info(f"Loaded data: {len(df)} records, {df['punch_code'].nunique()} punch codes")
        return df
    
    def create_enhanced_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Create features based on configuration - follows train_models2.py pattern"""
        enhanced_df = df.copy()
        
        # Add lag features
        if config.get('LAG_FEATURES', False):
            lags = config.get('lag_params', {}).get('lags', [1, 7, 14])
            enhanced_df = add_lag_features_by_group(enhanced_df, lags)
            
        # Add rolling features  
        if config.get('ROLLING_FEATURES', False):
            windows = config.get('rolling_params', {}).get('windows', [7, 14])
            enhanced_df = add_rolling_features_by_group(enhanced_df, windows)
            
        # Add date features
        if config.get('DATE_FEATURES', False):
            enhanced_df = add_date_features(enhanced_df)
            
        # Add cyclical features
        if config.get('CYCLICAL_FEATURES', False):
            enhanced_df = add_cyclical_features(enhanced_df)
            
        # Add trend features
        if config.get('TREND_FEATURES', False):
            enhanced_df = add_trend_features(enhanced_df)
            
        # Add pattern features
        if config.get('PATTERN_FEATURES', False):
            enhanced_df = add_pattern_features(enhanced_df)
            
        return enhanced_df
    
    def prepare_model_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target following train_models2.py pattern"""
        # Remove rows with missing values
        df_clean = df.dropna()
        
        # Define feature columns (exclude non-feature columns)
        exclude_cols = ['date', 'punch_code', 'total_hours']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean['total_hours']
        
        return X, y
    
    def evaluate_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model using time series cross-validation"""
        # Use TimeSeriesSplit for proper time series validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        mae_scores = []
        mse_scores = []
        r2_scores = []
        
        # RandomForest parameters following train_models2.py
        rf_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = RandomForestRegressor(**rf_params)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            mse_scores.append(mean_squared_error(y_val, y_pred))
            r2_scores.append(r2_score(y_val, y_pred))
        
        # Calculate statistics
        results = {
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'mse_mean': np.mean(mse_scores), 
            'mse_std': np.std(mse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'mape_mean': np.mean(mae_scores) / np.mean(y) * 100,
            'feature_count': len(X.columns)
        }
        
        return results
    
    def generate_feature_combinations(self) -> List[Dict]:
        """Generate all feature combinations to test"""
        feature_types = ['LAG_FEATURES', 'ROLLING_FEATURES', 'DATE_FEATURES', 
                        'CYCLICAL_FEATURES', 'TREND_FEATURES', 'PATTERN_FEATURES']
        
        # Parameter variations
        lag_variations = [
            {'lags': [1, 7]},
            {'lags': [1, 7, 14]},
            {'lags': [1, 2, 3, 7, 14]},
            {'lags': [1, 2, 3, 7, 14, 21, 28]}
        ]
        
        rolling_variations = [
            {'windows': [7]},
            {'windows': [7, 14]},
            {'windows': [3, 7, 14]},
            {'windows': [3, 7, 14, 30]}
        ]
        
        configurations = []
        
        # Test individual features first
        for feature in feature_types:
            config = {ft: False for ft in feature_types}
            config[feature] = True
            
            if feature == 'LAG_FEATURES':
                for lag_param in lag_variations:
                    config_copy = config.copy()
                    config_copy['lag_params'] = lag_param
                    configurations.append(config_copy)
            elif feature == 'ROLLING_FEATURES':
                for rolling_param in rolling_variations:
                    config_copy = config.copy()
                    config_copy['rolling_params'] = rolling_param
                    configurations.append(config_copy)
            else:
                configurations.append(config)
        
        # Test combinations of 2-3 features
        for combo_size in [2, 3]:
            for feature_combo in combinations(feature_types, combo_size):
                config = {ft: False for ft in feature_types}
                for feature in feature_combo:
                    config[feature] = True
                
                # Add parameter variations for LAG and ROLLING features
                if 'LAG_FEATURES' in feature_combo and 'ROLLING_FEATURES' in feature_combo:
                    for lag_param, rolling_param in product(lag_variations, rolling_variations):
                        config_copy = config.copy()
                        config_copy['lag_params'] = lag_param
                        config_copy['rolling_params'] = rolling_param
                        configurations.append(config_copy)
                elif 'LAG_FEATURES' in feature_combo:
                    for lag_param in lag_variations:
                        config_copy = config.copy()
                        config_copy['lag_params'] = lag_param
                        configurations.append(config_copy)
                elif 'ROLLING_FEATURES' in feature_combo:
                    for rolling_param in rolling_variations:
                        config_copy = config.copy()
                        config_copy['rolling_params'] = rolling_param
                        configurations.append(config_copy)
                else:
                    configurations.append(config)
        
        # Test all features combination
        all_features_config = {ft: True for ft in feature_types}
        for lag_param, rolling_param in product(lag_variations[:2], rolling_variations[:2]):
            config_copy = all_features_config.copy()
            config_copy['lag_params'] = lag_param
            config_copy['rolling_params'] = rolling_param
            configurations.append(config_copy)
        
        logger.info(f"Generated {len(configurations)} feature combinations to test")
        return configurations
    
    def run_feature_selection(self):
        """Run comprehensive feature selection"""
        logger.info("Starting feature selection process...")
        
        # Load data
        df = self.load_data()
        
        # Generate combinations
        configurations = self.generate_feature_combinations()
        
        total_configs = len(configurations)
        
        for i, config in enumerate(configurations, 1):
            logger.info(f"Testing configuration {i}/{total_configs}")
            logger.info(f"Config: {self._format_config_summary(config)}")
            
            try:
                start_time = time.time()
                
                # Create features
                enhanced_df = self.create_enhanced_features(df, config)
                
                # Prepare model data
                X, y = self.prepare_model_data(enhanced_df)
                
                if len(X) == 0:
                    logger.warning("No valid data after feature creation, skipping...")
                    continue
                
                # Evaluate performance
                performance = self.evaluate_model_performance(X, y)
                
                # Record results
                result = {
                    'config_id': i,
                    'config': config,
                    'performance': performance,
                    'training_time': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                
                # Check if this is the best configuration
                if performance['mae_mean'] < self.best_score:
                    self.best_score = performance['mae_mean']
                    self.best_config = config.copy()
                
                # Log performance
                logger.info(f"MAE: {performance['mae_mean']:.4f} ± {performance['mae_std']:.4f}")
                logger.info(f"R²: {performance['r2_mean']:.4f} ± {performance['r2_std']:.4f}")
                logger.info(f"MAPE: {performance['mape_mean']:.2f}%")
                logger.info(f"Features: {performance['feature_count']}")
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"Error testing configuration {i}: {str(e)}")
                continue
        
        # Save results and generate report
        self.save_results()
        self.generate_report()
        
    def _format_config_summary(self, config: Dict) -> str:
        """Format configuration for logging"""
        active_features = [k for k, v in config.items() if v is True and k.endswith('_FEATURES')]
        summary = f"Features: {', '.join(active_features)}"
        
        if 'lag_params' in config:
            summary += f" | Lags: {config['lag_params']['lags']}"
        if 'rolling_params' in config:
            summary += f" | Windows: {config['rolling_params']['windows']}"
            
        return summary
    
    def save_results(self):
        """Save detailed results to JSON file"""
        results_file = f"feature_selection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if not self.results:
            logger.error("No results to analyze")
            return
        
        # Sort results by MAE
        sorted_results = sorted(self.results, key=lambda x: x['performance']['mae_mean'])
        
        logger.info("=" * 80)
        logger.info("FEATURE SELECTION ANALYSIS REPORT")
        logger.info("=" * 80)
        
        # Best performing configurations
        logger.info("\n🏆 TOP 5 BEST PERFORMING CONFIGURATIONS:")
        logger.info("-" * 50)
        
        for i, result in enumerate(sorted_results[:5], 1):
            perf = result['performance']
            config = result['config']
            
            logger.info(f"\n#{i} CONFIGURATION (ID: {result['config_id']})")
            logger.info(f"   Features: {self._format_config_summary(config)}")
            logger.info(f"   MAE: {perf['mae_mean']:.4f} ± {perf['mae_std']:.4f}")
            logger.info(f"   R²: {perf['r2_mean']:.4f} ± {perf['r2_std']:.4f}")
            logger.info(f"   MAPE: {perf['mape_mean']:.2f}%")
            logger.info(f"   Feature Count: {perf['feature_count']}")
            logger.info(f"   Training Time: {result['training_time']:.2f}s")
        
        # Feature type analysis
        self._analyze_feature_types()
        
        # Parameter analysis
        self._analyze_parameters()
        
        # Final recommendation
        logger.info("\n🎯 RECOMMENDED CONFIGURATION FOR train_models2.py:")
        logger.info("-" * 50)
        best_result = sorted_results[0]
        logger.info(f"Configuration ID: {best_result['config_id']}")
        logger.info(f"Expected MAE: {best_result['performance']['mae_mean']:.4f}")
        logger.info(f"Expected R²: {best_result['performance']['r2_mean']:.4f}")
        logger.info(f"Expected MAPE: {best_result['performance']['mape_mean']:.2f}%")
        
        # Generate config.py update
        self._generate_config_update(best_result['config'])
        
    def _analyze_feature_types(self):
        """Analyze performance by feature type"""
        logger.info("\n📊 FEATURE TYPE ANALYSIS:")
        logger.info("-" * 30)
        
        feature_types = ['LAG_FEATURES', 'ROLLING_FEATURES', 'DATE_FEATURES', 
                        'CYCLICAL_FEATURES', 'TREND_FEATURES', 'PATTERN_FEATURES']
        
        for feature_type in feature_types:
            with_feature = [r for r in self.results if r['config'].get(feature_type, False)]
            without_feature = [r for r in self.results if not r['config'].get(feature_type, False)]
            
            if with_feature and without_feature:
                avg_mae_with = np.mean([r['performance']['mae_mean'] for r in with_feature])
                avg_mae_without = np.mean([r['performance']['mae_mean'] for r in without_feature])
                improvement = ((avg_mae_without - avg_mae_with) / avg_mae_without) * 100
                
                logger.info(f"{feature_type}: {improvement:+.2f}% impact on MAE")
    
    def _analyze_parameters(self):
        """Analyze parameter impact"""
        logger.info("\n⚙️ PARAMETER ANALYSIS:")
        logger.info("-" * 25)
        
        # Lag parameter analysis
        lag_configs = [r for r in self.results if 'lag_params' in r['config']]
        if lag_configs:
            lag_analysis = {}
            for result in lag_configs:
                lags = str(result['config']['lag_params']['lags'])
                if lags not in lag_analysis:
                    lag_analysis[lags] = []
                lag_analysis[lags].append(result['performance']['mae_mean'])
            
            logger.info("LAG PARAMETERS:")
            for lags, mae_scores in lag_analysis.items():
                logger.info(f"  {lags}: Avg MAE = {np.mean(mae_scores):.4f}")
        
        # Rolling parameter analysis  
        rolling_configs = [r for r in self.results if 'rolling_params' in r['config']]
        if rolling_configs:
            rolling_analysis = {}
            for result in rolling_configs:
                windows = str(result['config']['rolling_params']['windows'])
                if windows not in rolling_analysis:
                    rolling_analysis[windows] = []
                rolling_analysis[windows].append(result['performance']['mae_mean'])
            
            logger.info("ROLLING PARAMETERS:")
            for windows, mae_scores in rolling_analysis.items():
                logger.info(f"  {windows}: Avg MAE = {np.mean(mae_scores):.4f}")
    
    def _generate_config_update(self, best_config: Dict):
        """Generate config.py update instructions"""
        logger.info("\n📝 CONFIG.PY UPDATE INSTRUCTIONS:")
        logger.info("-" * 35)
        logger.info("Update your config.py with these settings:")
        logger.info("")
        
        feature_types = ['LAG_FEATURES', 'ROLLING_FEATURES', 'DATE_FEATURES', 
                        'CYCLICAL_FEATURES', 'TREND_FEATURES', 'PATTERN_FEATURES']
        
        for feature_type in feature_types:
            value = best_config.get(feature_type, False)
            logger.info(f"{feature_type} = {value}")
        
        if 'lag_params' in best_config:
            logger.info(f"LAG_PERIODS = {best_config['lag_params']['lags']}")
            
        if 'rolling_params' in best_config:
            logger.info(f"ROLLING_WINDOWS = {best_config['rolling_params']['windows']}")


def main():
    """Main execution function"""
    
    # Configuration
    DATA_PATH = "data/processed_workforce_data.csv"  # Update with your data path
    TARGET_PUNCH_CODES = [202, 203, 206, 209, 210, 211, 213, 214, 215, 217]
    
    # Initialize selector
    selector = FeatureCombinationSelector(
        data_path=DATA_PATH,
        target_punch_codes=TARGET_PUNCH_CODES
    )
    
    # Run feature selection
    selector.run_feature_selection()
    
    logger.info("Feature selection completed!")


if __name__ == "__main__":
    main()