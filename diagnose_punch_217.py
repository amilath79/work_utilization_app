import pandas as pd
import numpy as np
from config import MODELS_DIR
import os
from utils.feature_engineering import EnhancedFeatureTransformer
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def deep_diagnose_217():
    """Deep diagnosis of punch code 217 issues"""
    
    print("🔍 DEEP DIAGNOSIS: Punch Code 217")
    print("=" * 50)
    
    # Load data
    df = pd.read_pickle(os.path.join(MODELS_DIR, 'enhanced_training_data.pkl'))
    code_217_data = df[df['WorkType'] == '217'].copy()
    code_217_data = code_217_data.sort_values('Date')
    
    print(f"📊 Basic Info:")
    print(f"   Records: {len(code_217_data)}")
    print(f"   Date range: {code_217_data['Date'].min()} to {code_217_data['Date'].max()}")
    
    # 1. CHECK FOR TEMPORAL PATTERNS
    print(f"\n📈 Temporal Pattern Analysis:")
    code_217_data['Year'] = code_217_data['Date'].dt.year
    code_217_data['Month'] = code_217_data['Date'].dt.month
    code_217_data['DayOfWeek'] = code_217_data['Date'].dt.dayofweek
    
    yearly_stats = code_217_data.groupby('Year')['Hours'].agg(['count', 'mean', 'std'])
    print("   Year-by-Year Pattern:")
    print(yearly_stats.round(2))
    
    # Check for data shifts
    recent_data = code_217_data[code_217_data['Date'] >= '2023-01-01']
    older_data = code_217_data[code_217_data['Date'] < '2023-01-01']
    
    if len(recent_data) > 50 and len(older_data) > 50:
        recent_mean = recent_data['Hours'].mean()
        older_mean = older_data['Hours'].mean()
        shift_pct = (recent_mean - older_mean) / older_mean * 100
        
        print(f"\n🔄 Data Shift Analysis:")
        print(f"   Older data (pre-2023): {older_mean:.1f} hours avg")
        print(f"   Recent data (2023+): {recent_mean:.1f} hours avg")
        print(f"   Shift: {shift_pct:+.1f}%")
        
        if abs(shift_pct) > 50:
            print(f"   ⚠️  MAJOR DATA SHIFT DETECTED! ({shift_pct:+.1f}%)")
    
    # 2. CHECK FEATURE ENGINEERING ISSUES
    print(f"\n🔧 Feature Engineering Test:")
    try:
        transformer = EnhancedFeatureTransformer()
        transformed_data = transformer.fit_transform(code_217_data)
        
        print(f"   ✅ Feature engineering successful: {len(transformed_data.columns)} features")
        
        # Check for NaN/Inf issues
        nan_counts = transformed_data.isnull().sum()
        problematic_features = nan_counts[nan_counts > len(transformed_data) * 0.8]  # >80% missing
        
        if len(problematic_features) > 0:
            print(f"   ⚠️  High missing value features:")
            for feat, count in problematic_features.items():
                print(f"      {feat}: {count}/{len(transformed_data)} missing")
        
        # Check for constant features
        numeric_cols = transformed_data.select_dtypes(include=[np.number]).columns
        constant_features = []
        for col in numeric_cols:
            if transformed_data[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"   ⚠️  Constant features (no variance): {len(constant_features)}")
            print(f"      {constant_features[:5]}...")  # Show first 5
            
    except Exception as e:
        print(f"   ❌ Feature engineering failed: {e}")
        return
    
    # 3. CHECK LAG FEATURE AVAILABILITY
    print(f"\n📅 Lag Feature Analysis:")
    
    # Check yearly lag coverage
    yearly_lag_365 = code_217_data.groupby(code_217_data['Date'].dt.year)['Hours'].count()
    print("   Records per year (for yearly lags):")
    print(yearly_lag_365.to_dict())
    
    # Years with <50 records might cause yearly lag issues
    low_coverage_years = yearly_lag_365[yearly_lag_365 < 50]
    if len(low_coverage_years) > 0:
        print(f"   ⚠️  Low coverage years: {low_coverage_years.to_dict()}")
    
    # 4. MODEL-SPECIFIC DIAGNOSTICS
    print(f"\n🤖 Model Training Simulation:")
    
    try:
        # Simulate the training split (last 20% for testing)
        split_idx = int(len(transformed_data) * 0.8)
        
        X_train = transformed_data.iloc[:split_idx]
        y_train = X_train['Hours'] if 'Hours' in X_train.columns else code_217_data.iloc[:split_idx]['Hours']
        
        X_test = transformed_data.iloc[split_idx:]
        y_test = X_test['Hours'] if 'Hours' in X_test.columns else code_217_data.iloc[split_idx:]['Hours']
        
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        print(f"   Train Hours - Mean: {y_train.mean():.1f}, Std: {y_train.std():.1f}")
        print(f"   Test Hours - Mean: {y_test.mean():.1f}, Std: {y_test.std():.1f}")
        
        # Check for train/test distribution mismatch
        train_test_diff = abs(y_train.mean() - y_test.mean()) / y_train.mean() * 100
        if train_test_diff > 30:
            print(f"   🚨 TRAIN/TEST MISMATCH: {train_test_diff:.1f}% difference in means")
            print(f"      This explains the poor R² score!")
        
        # Check test set for outliers
        test_outliers = (y_test > y_test.quantile(0.95)).sum()
        print(f"   Test set outliers (>95th percentile): {test_outliers}/{len(y_test)}")
        
        if test_outliers > len(y_test) * 0.1:  # >10% outliers
            print(f"   ⚠️  High outlier concentration in test set")
            
    except Exception as e:
        print(f"   ❌ Model simulation failed: {e}")
    
    # 5. RECOMMENDATIONS
    print(f"\n💡 RECOMMENDATIONS:")
    
    if 'train_test_diff' in locals() and train_test_diff > 30:
        print("   🔧 PRIORITY 1: Fix train/test distribution mismatch")
        print("      - Use stratified sampling by Hours quartiles")
        print("      - Or use time-series cross-validation")
    
    if len(low_coverage_years) > 2:
        print("   🔧 PRIORITY 2: Address yearly lag issues")
        print("      - Consider removing yearly features for this punch code")
        print("      - Or use shorter lags (90-day, 180-day)")
    
    if len(constant_features) > 5:
        print("   🔧 PRIORITY 3: Remove constant features")
        print("      - Filter out features with zero variance")
    
    # SPECIFIC FIX SUGGESTION
    print(f"\n🛠️  IMMEDIATE FIX FOR PUNCH CODE 217:")
    print("   1. Retrain with stratified sampling")
    print("   2. Remove yearly lag features for this code")
    print("   3. Use only recent data (2022+)")
    print("   4. Apply stronger regularization")

if __name__ == "__main__":
    deep_diagnose_217()