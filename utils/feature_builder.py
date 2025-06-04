"""
Centralized feature builder that works for both training and prediction
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from config import (
    FEATURE_TIERS, BASIC_FEATURES, INTERMEDIATE_FEATURES, ADVANCED_FEATURES,
    FEATURE_AVAILABILITY
)

logger = logging.getLogger(__name__)

class FeatureBuilder:
    """
    Builds consistent feature sets for both training and prediction
    """
    
    def __init__(self, data_columns=None, data_length=0):
        """
        Initialize feature builder
        
        Parameters:
        -----------
        data_columns : list
            Available columns in the dataset
        data_length : int  
            Number of rows in the dataset
        """
        self.data_columns = data_columns or []
        self.data_length = data_length
        self.available_features = self._detect_available_features()
        
    def _detect_available_features(self):
        """Detect which feature tiers can be used based on available data"""
        available = {'BASIC': False, 'INTERMEDIATE': False, 'ADVANCED': False}
        
        # Check data length requirements
        min_reqs = FEATURE_AVAILABILITY['MIN_DATA_REQUIREMENTS']
        
        if self.data_length >= min_reqs['BASIC']:
            available['BASIC'] = True
            
        if self.data_length >= min_reqs['INTERMEDIATE']:
            available['INTERMEDIATE'] = True
            
        if self.data_length >= min_reqs['ADVANCED']:
            available['ADVANCED'] = True
            
        # Check productivity feature availability
        productivity_cols = FEATURE_AVAILABILITY['PRODUCTIVITY_REQUIRED_COLUMNS']
        has_productivity = all(col in self.data_columns for col in productivity_cols)
        
        if not has_productivity and available['ADVANCED']:
            logger.warning("Advanced productivity features disabled - missing required columns")
            
        available['HAS_PRODUCTIVITY'] = has_productivity
        
        return available
        
    def get_feature_lists(self):
        """
        Get comprehensive feature lists based on enabled tiers
        
        Returns:
        --------
        tuple: (numeric_features, categorical_features)
        """
        numeric_features = []
        categorical_features = []
        
        # Add features from each enabled tier
        if FEATURE_TIERS['BASIC'] and self.available_features['BASIC']:
            num_feat, cat_feat = self._build_basic_features()
            numeric_features.extend(num_feat)
            categorical_features.extend(cat_feat)
            
        if FEATURE_TIERS['INTERMEDIATE'] and self.available_features['INTERMEDIATE']:
            num_feat, cat_feat = self._build_intermediate_features()
            numeric_features.extend(num_feat)
            categorical_features.extend(cat_feat)
            
        if FEATURE_TIERS['ADVANCED'] and self.available_features['ADVANCED']:
            num_feat, cat_feat = self._build_advanced_features()
            numeric_features.extend(num_feat)
            categorical_features.extend(cat_feat)
        
        # Remove duplicates while preserving order
        numeric_features = list(dict.fromkeys(numeric_features))
        categorical_features = list(dict.fromkeys(categorical_features))
        
        logger.info(f"Built feature set: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
        return numeric_features, categorical_features
    
    def _build_basic_features(self):
        """Build basic tier features"""
        numeric = []
        categorical = []
        
        # Date features
        categorical.extend(BASIC_FEATURES['DATE_FEATURES']['categorical'])
        numeric.extend(BASIC_FEATURES['DATE_FEATURES']['numeric'])
        
        # Lag features
        for base_feature, lags in BASIC_FEATURES['LAG_FEATURES'].items():
            for lag in lags:
                numeric.append(f'{base_feature}_lag_{lag}')
                
        # Rolling features
        for base_feature, config in BASIC_FEATURES['ROLLING_FEATURES'].items():
            for window in config['windows']:
                for func in config['functions']:
                    numeric.append(f'{base_feature}_rolling_{func}_{window}')
        
        return numeric, categorical
    
    def _build_intermediate_features(self):
        """Build intermediate tier features"""
        numeric = []
        categorical = []
        
        # Date features
        categorical.extend(INTERMEDIATE_FEATURES['DATE_FEATURES']['categorical'])
        numeric.extend(INTERMEDIATE_FEATURES['DATE_FEATURES']['numeric'])
        
        # Lag features
        for base_feature, lags in INTERMEDIATE_FEATURES['LAG_FEATURES'].items():
            for lag in lags:
                numeric.append(f'{base_feature}_lag_{lag}')
        
        # Rolling features
        for base_feature, config in INTERMEDIATE_FEATURES['ROLLING_FEATURES'].items():
            for window in config['windows']:
                for func in config['functions']:
                    numeric.append(f'{base_feature}_rolling_{func}_{window}')
                    
            # Extended stats for first window
            if 'extended_stats' in config:
                ext_config = config['extended_stats']
                for func in ext_config['functions']:
                    numeric.append(f'{base_feature}_rolling_{func}_{ext_config["window"]}')
        
        # Pattern features
        numeric.extend(INTERMEDIATE_FEATURES['PATTERN_FEATURES'])
        
        # Trend features (add if required lags exist)
        for trend_name, base_feature, lag1, lag2 in INTERMEDIATE_FEATURES['TREND_FEATURES']:
            numeric.append(trend_name)
            
        return numeric, categorical
    
    def _build_advanced_features(self):
        """Build advanced tier features"""
        numeric = []
        categorical = []
        
        # Date features
        categorical.extend(ADVANCED_FEATURES['DATE_FEATURES']['categorical'])
        numeric.extend(ADVANCED_FEATURES['DATE_FEATURES']['numeric'])
        
        # Lag features
        for base_feature, lags in ADVANCED_FEATURES['LAG_FEATURES'].items():
            for lag in lags:
                numeric.append(f'{base_feature}_lag_{lag}')
        
        # Rolling features
        for base_feature, config in ADVANCED_FEATURES['ROLLING_FEATURES'].items():
            for window in config['windows']:
                for func in config['functions']:
                    numeric.append(f'{base_feature}_rolling_{func}_{window}')
        
        # Productivity features (if available)
        if self.available_features['HAS_PRODUCTIVITY']:
            prod_numeric, prod_categorical = self._build_productivity_features()
            numeric.extend(prod_numeric)
            categorical.extend(prod_categorical)
        
        # Business features
        numeric.extend(ADVANCED_FEATURES['BUSINESS_FEATURES'])
        
        return numeric, categorical
    
    def _build_productivity_features(self):
        """Build productivity features"""
        numeric = []
        categorical = []
        
        prod_config = ADVANCED_FEATURES['PRODUCTIVITY_FEATURES']
        
        # Productivity lag features
        for base_feature, lags in prod_config['LAG_FEATURES'].items():
            for lag in lags:
                numeric.append(f'{base_feature}_lag_{lag}')
        
        # Productivity rolling features
        for base_feature, config in prod_config['ROLLING_FEATURES'].items():
            for window in config['windows']:
                for func in config['functions']:
                    numeric.append(f'{base_feature}_rolling_{func}_{window}')
        
        # Derived features
        numeric.extend(prod_config['DERIVED_FEATURES'])
        
        # Pattern features
        numeric.extend(prod_config['PATTERN_FEATURES'])
        
        # Trend features
        for trend_name, base_feature, lag1, lag2 in prod_config['TREND_FEATURES']:
            numeric.append(trend_name)
        
        return numeric, categorical

# Global function for easy access
def get_feature_lists_for_data(data_columns=None, data_length=0):
    """
    Get feature lists based on available data
    
    Parameters:
    -----------
    data_columns : list
        Available columns in dataset
    data_length : int
        Number of rows in dataset
        
    Returns:
    --------
    tuple: (numeric_features, categorical_features)
    """
    builder = FeatureBuilder(data_columns, data_length)
    return builder.get_feature_lists()