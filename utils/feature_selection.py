# File: utils/feature_selection.py (NEW FILE)

"""
Feature selection utilities to prevent overfitting in LightGBM models
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.inspection import permutation_importance
import logging

logger = logging.getLogger(__name__)

def select_features_by_importance(model, X, y, feature_names, top_k=30):
    """
    Select top K features based on LightGBM feature importance
    
    Parameters:
    -----------
    model : LGBMRegressor
        Trained LightGBM model
    X : array-like
        Feature matrix
    y : array-like
        Target values
    feature_names : list
        Feature names
    top_k : int
        Number of top features to select
        
    Returns:
    --------
    list
        Selected feature names
    """
    try:
        # Get feature importances
        importances = model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top K features
        selected_features = importance_df.head(top_k)['feature'].tolist()
        
        # Always include essential features
        essential_features = ['Date', 'WorkType']
        for feat in essential_features:
            if feat in feature_names and feat not in selected_features:
                selected_features.append(feat)
        
        logger.info(f"Selected {len(selected_features)} features from {len(feature_names)}")
        logger.info(f"Top 10 features: {importance_df.head(10)['feature'].tolist()}")
        
        return selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        return feature_names  # Return all features if selection fails

def reduce_features_gradually(X, y, feature_names, initial_features=50, min_features=20):
    """
    Gradually reduce features to find optimal set
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    feature_names : list
        Feature names
    initial_features : int
        Starting number of features
    min_features : int
        Minimum number of features to keep
        
    Returns:
    --------
    dict
        Optimal feature configuration
    """
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import cross_val_score
    
    best_score = -np.inf
    best_n_features = initial_features
    best_features = feature_names
    
    # Test different feature counts
    for n_features in range(initial_features, min_features - 1, -5):
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, len(feature_names)))
        X_selected = selector.fit_transform(X, y)
        selected_mask = selector.get_support()
        selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
        
        # Quick model evaluation
        model = LGBMRegressor(
            n_estimators=100,
            num_leaves=20,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        )
        
        # Use negative MAE for scoring (higher is better)
        scores = cross_val_score(model, X_selected, y, cv=3, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
        avg_score = scores.mean()
        
        logger.info(f"Features: {n_features}, CV MAE: {-avg_score:.3f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_n_features = n_features
            best_features = selected_features
    
    logger.info(f"Optimal feature count: {best_n_features} (MAE: {-best_score:.3f})")
    
    return {
        'n_features': best_n_features,
        'features': best_features,
        'mae': -best_score
    }

# Add to train_models2.py imports:
# from utils.feature_selection import select_features_by_importance, reduce_features_gradually

# IMPACT: Reduces overfitting by using only the most predictive features


from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.features]