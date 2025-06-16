#!/usr/bin/env python3
"""
Test script to verify data setup before running feature selection
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_data_setup():
    """Test data file and structure"""
    
    print("🔍 TESTING DATA SETUP FOR FEATURE SELECTION")
    print("=" * 60)
    
    # Try to import configuration
    try:
        from feature_selector_config import DATA_PATH, TARGET_PUNCH_CODES
        print(f"✅ Configuration imported successfully")
        print(f"📁 Data path: {DATA_PATH}")
        print(f"🎯 Target punch codes: {TARGET_PUNCH_CODES}")
    except ImportError:
        DATA_PATH = "data/processed_workforce_data.csv"
        TARGET_PUNCH_CODES = [202, 203, 206, 209, 210, 211, 213, 214, 215, 217]
        print(f"⚠️  Using default configuration")
        print(f"📁 Data path: {DATA_PATH}")
        print(f"🎯 Target punch codes: {TARGET_PUNCH_CODES}")
    
    print("\n" + "-" * 60)
    
    # Check if data file exists
    if not Path(DATA_PATH).exists():
        print(f"❌ Data file not found: {DATA_PATH}")
        print(f"\n🔧 SOLUTIONS:")
        print(f"1. Update DATA_PATH in feature_selector_config.py")
        print(f"2. Make sure your data file exists")
        print(f"3. Check the file path and name")
        return False
    else:
        print(f"✅ Data file found: {DATA_PATH}")
    
    # Load and examine data
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data loaded successfully")
        print(f"📊 Shape: {df.shape}")
        
        print(f"\n📋 COLUMN ANALYSIS:")
        print(f"-" * 25)
        print(f"Columns found: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['date', 'punch_code', 'total_hours']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            
            # Try to suggest mappings
            print(f"\n🔧 COLUMN MAPPING SUGGESTIONS:")
            for missing_col in missing_cols:
                if missing_col == 'date':
                    date_like = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                    if date_like:
                        print(f"  For 'date': Consider {date_like}")
                elif missing_col == 'punch_code':
                    code_like = [col for col in df.columns if 'code' in col.lower() or 'type' in col.lower() or 'id' in col.lower()]
                    if code_like:
                        print(f"  For 'punch_code': Consider {code_like}")
                elif missing_col == 'total_hours':
                    hours_like = [col for col in df.columns if 'hour' in col.lower() or 'time' in col.lower()]
                    if hours_like:
                        print(f"  For 'total_hours': Consider {hours_like}")
        else:
            print(f"✅ All required columns found: {required_cols}")
        
        # Check data types
        print(f"\n📊 DATA TYPES:")
        print(f"-" * 15)
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        # Check for target punch codes
        if 'punch_code' in df.columns:
            unique_codes = df['punch_code'].unique()
            print(f"\n🎯 PUNCH CODE ANALYSIS:")
            print(f"-" * 25)
            print(f"Unique punch codes in data: {sorted(unique_codes)}")
            
            found_codes = [code for code in TARGET_PUNCH_CODES if code in unique_codes]
            missing_codes = [code for code in TARGET_PUNCH_CODES if code not in unique_codes]
            
            print(f"✅ Target codes found: {found_codes}")
            if missing_codes:
                print(f"⚠️  Target codes missing: {missing_codes}")
            
            # Check data distribution
            filtered_df = df[df['punch_code'].isin(TARGET_PUNCH_CODES)]
            print(f"📊 Records after filtering: {len(filtered_df)} / {len(df)}")
            
            if len(filtered_df) == 0:
                print(f"❌ No data remains after filtering for target punch codes!")
                return False
        
        # Check date column
        if 'date' in df.columns:
            print(f"\n📅 DATE ANALYSIS:")
            print(f"-" * 18)
            try:
                df['date'] = pd.to_datetime(df['date'])
                print(f"✅ Date conversion successful")
                print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"Total days: {(df['date'].max() - df['date'].min()).days}")
            except Exception as e:
                print(f"⚠️  Date conversion issue: {e}")
        
        # Check for missing values
        print(f"\n🔍 MISSING VALUES:")
        print(f"-" * 20)
        missing_summary = df.isnull().sum()
        if missing_summary.sum() == 0:
            print(f"✅ No missing values found")
        else:
            for col, missing_count in missing_summary.items():
                if missing_count > 0:
                    pct = (missing_count / len(df)) * 100
                    print(f"  {col}: {missing_count} ({pct:.1f}%)")
        
        # Sample data preview
        print(f"\n📋 SAMPLE DATA (first 3 rows):")
        print(f"-" * 35)
        print(df.head(3).to_string())
        
        print(f"\n✅ DATA SETUP VERIFICATION COMPLETE!")
        print(f"🚀 Ready to run: python best_feature_selector.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functions"""
    
    print(f"\n🔧 TESTING FEATURE ENGINEERING FUNCTIONS:")
    print(f"-" * 45)
    
    try:
        from utils.feature_engineering import create_lag_features, EnhancedFeatureTransformer
        print(f"✅ Successfully imported create_lag_features and EnhancedFeatureTransformer")
        
        # Test with sample data
        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'punch_code': [202, 203] * 5,
            'total_hours': [8.0, 7.5, 8.2, 7.8, 8.1, 7.9, 8.0, 7.7, 8.3, 7.6]
        })
        
        print(f"📊 Testing with sample data...")
        
        # Test create_lag_features
        try:
            result = create_lag_features(
                df=sample_data,
                group_col='punch_code',
                target_col='total_hours',
                lag_days=[1, 2],
                rolling_windows=[3]
            )
            print(f"✅ create_lag_features working correctly")
            print(f"   Created {result.shape[1] - sample_data.shape[1]} new features")
            
        except Exception as e:
            print(f"⚠️  create_lag_features issue: {e}")
        
        # Test EnhancedFeatureTransformer
        try:
            transformer = EnhancedFeatureTransformer()
            print(f"✅ EnhancedFeatureTransformer initialized successfully")
            
        except Exception as e:
            print(f"⚠️  EnhancedFeatureTransformer issue: {e}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    
    print("🧪 FEATURE SELECTION SETUP TESTING")
    print("=" * 50)
    
    # Test data setup
    data_ok = test_data_setup()
    
    # Test feature engineering
    fe_ok = test_feature_engineering()
    
    print(f"\n" + "=" * 50)
    print(f"📊 SETUP TEST SUMMARY:")
    print(f"-" * 25)
    print(f"Data setup: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"Feature engineering: {'✅ PASS' if fe_ok else '❌ FAIL'}")
    
    if data_ok and fe_ok:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"🚀 Ready to run feature selection:")
        print(f"   python best_feature_selector.py")
    else:
        print(f"\n⚠️  PLEASE FIX ISSUES BEFORE RUNNING FEATURE SELECTION")
        
        if not data_ok:
            print(f"   - Fix data file path and structure")
        if not fe_ok:
            print(f"   - Fix feature engineering imports")

if __name__ == "__main__":
    main()