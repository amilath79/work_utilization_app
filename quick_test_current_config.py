"""
Quick test to verify current config works before running full optimization
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import config
from utils.sql_data_connector import extract_sql_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data():
    """Load training data from SQL"""
    try:
        query = """
        SELECT Date, PunchCode as WorkType, Hours, SystemHours, 
		CASE WHEN PunchCode IN (206, 213) THEN NoRows
		ELSE Quantity END as Quantity, 
		SystemKPI
        FROM WorkUtilizationData 
        WHERE PunchCode IN ('202', '203', '206', '209', '210', '211', '213', '214', '215', '217') 
		AND Hours > 0 
        AND SystemHours > 0 
        AND NoRows > 0
        AND Date < '2025-05-06'
        ORDER BY Date
        """
        
        df = extract_sql_data(
            server=config.SQL_SERVER,
            database=config.SQL_DATABASE,
            query=query,
            trusted_connection=config.SQL_TRUSTED_CONNECTION
        )
        
        if df is None or df.empty:
            return None
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['WorkType'] = df['WorkType'].astype(str)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

def create_simple_features(df, punch_code):
    """
    Create simple features without using EnhancedFeatureTransformer
    To test if basic feature engineering works
    """
    try:
        # Filter to specific punch code
        data = df[df['WorkType'] == punch_code].copy()
        data = data.sort_values('Date')
        
        logger.info(f"Creating features for {punch_code} with {len(data)} records")
        
        # Basic date features
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['Month'] = data['Date'].dt.month
        data['IsWeekend'] = (data['Date'].dt.dayofweek >= 5).astype(int)
        
        # Simple lag features using current config
        for lag in config.ESSENTIAL_LAGS[:3]:  # Use first 3 lags only
            data[f'Hours_lag_{lag}'] = data['Hours'].shift(lag)
            
        # Simple rolling features
        for window in config.ESSENTIAL_WINDOWS[:2]:  # Use first 2 windows only
            data[f'Hours_rolling_mean_{window}'] = data['Hours'].rolling(window, min_periods=1).mean()
        
        # Drop rows with NaN (due to lags)
        data = data.dropna()
        
        if len(data) < 20:
            logger.warning(f"Too few records after feature creation: {len(data)}")
            return None, None
            
        # Prepare features and target
        feature_cols = [col for col in data.columns if col.endswith(('_lag_1', '_lag_2', '_lag_7', '_rolling_mean_7', '_rolling_mean_14', 'DayOfWeek', 'Month', 'IsWeekend'))]
        
        if len(feature_cols) == 0:
            logger.warning("No feature columns found")
            return None, None
            
        X = data[feature_cols].fillna(0)
        y = data['Hours']
        
        logger.info(f"Created {len(feature_cols)} features for {len(X)} records")
        logger.info(f"Features: {feature_cols}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error creating features: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def test_current_config():
    """Test current configuration with simple feature engineering"""
    logger.info("🧪 TESTING CURRENT CONFIG WITH SIMPLE FEATURES")
    
    try:
        # Load data
        df = load_training_data()
        if df is None:
            logger.error("Failed to load data")
            return
            
        logger.info(f"Loaded {len(df)} records")
        
        # Test both punch codes
        results = {}
        
        for punch_code in ['206', '213']:
            logger.info(f"\n🔍 Testing {punch_code}...")
            
            X, y = create_simple_features(df, punch_code)
            
            if X is None or y is None:
                logger.error(f"Failed to create features for {punch_code}")
                continue
                
            # Simple model test
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', RandomForestRegressor(
                    n_estimators=50,  # Fewer trees for quick test
                    max_depth=6,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            # Quick 3-fold CV
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_absolute_error')
            mae_scores = -cv_scores
            avg_mae = np.mean(mae_scores)
            
            r2_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
            avg_r2 = np.mean(r2_scores)
            
            results[punch_code] = {'mae': avg_mae, 'r2': avg_r2}
            
            # Status
            mae_status = "✅" if avg_mae < 0.5 else "❌"
            r2_status = "✅" if avg_r2 > 0.85 else "❌"
            
            logger.info(f"   MAE: {avg_mae:.3f} {mae_status}")
            logger.info(f"   R²:  {avg_r2:.3f} {r2_status}")
        
        # Summary
        if results:
            overall_mae = np.mean([r['mae'] for r in results.values()])
            overall_r2 = np.mean([r['r2'] for r in results.values()])
            
            print("\n" + "="*50)
            print("📊 SIMPLE FEATURE TEST RESULTS")
            print("="*50)
            print(f"Overall MAE: {overall_mae:.3f} (Target: < 0.5)")
            print(f"Overall R²:  {overall_r2:.3f} (Target: > 0.85)")
            
            if overall_mae < 0.5 and overall_r2 > 0.85:
                print("✅ CURRENT CONFIG LOOKS GOOD - Proceed with train_models2.py")
                print("🚀 Recommendation: Skip optimization, run 'python train_models2.py'")
            else:
                print("⚠️ Current config needs optimization")
                print("🔧 Recommendation: Fix optimization script or adjust config manually")
            
            print("="*50)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_current_config()