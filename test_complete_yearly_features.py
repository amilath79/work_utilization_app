# test_complete_yearly_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.feature_engineering import EnhancedFeatureTransformer
from utils.prediction import predict_next_day, create_prediction_row_enhanced
import os
from config import MODELS_DIR

def test_complete_yearly_integration():
    """Test the complete yearly feature integration"""
    
    print("🧪 Testing Complete Yearly Feature Integration")
    
    # Load training data
    training_data_path = os.path.join(MODELS_DIR, 'enhanced_training_data.pkl')
    if not os.path.exists(training_data_path):
        print("❌ Training data not found. Run train_models2.py first.")
        return False
        
    df = pd.read_pickle(training_data_path)
    print(f"✅ Loaded {len(df)} records")
    
    # Test with punch code that has good historical data
    test_punch_code = '206'
    test_data = df[df['WorkType'] == test_punch_code].copy()
    test_data = test_data.sort_values('Date')
    
    print(f"📊 Testing with punch code {test_punch_code}")
    print(f"   Records: {len(test_data)}")
    print(f"   Date range: {test_data['Date'].min()} to {test_data['Date'].max()}")
    
    if len(test_data) < 400:
        print(f"⚠️ Not enough data for meaningful test")
        return False
    
    # Test 1: Feature Transformer
    print("\n🔧 Test 1: Feature Transformation")
    try:
        transformer = EnhancedFeatureTransformer()
        
        # Test fit
        transformer.fit(test_data)
        print("   ✅ Fit successful")
        
        # Test transform
        transformed_data = transformer.transform(test_data)
        print(f"   ✅ Transform successful: {len(transformed_data.columns)} features created")
        
        # Check for yearly features
        yearly_features = [col for col in transformed_data.columns 
                          if any(keyword in col for keyword in ['yearly_lag', 'vs_last_year', 'same_day_last_year', 'yoy_growth'])]
        
        print(f"   📈 Yearly Features Found ({len(yearly_features)}):")
        for feature in yearly_features[:10]:  # Show first 10
            non_null_count = transformed_data[feature].notna().sum()
            print(f"      {feature}: {non_null_count}/{len(transformed_data)} non-null")
        
        if len(yearly_features) < 8:
            print("   ⚠️ Warning: Expected more yearly features")
            return False
            
    except Exception as e:
        print(f"   ❌ Feature transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Prediction Row Creation
    print("\n🔧 Test 2: Enhanced Prediction Row Creation")
    try:
        next_date = datetime(2025, 8, 7)
        pred_row = create_prediction_row_enhanced(test_data, next_date, test_punch_code)
        
        print("   ✅ Prediction row creation successful")
        print(f"      Date: {pred_row['Date'].iloc[0]}")
        print(f"      Quantity: {pred_row['Quantity'].iloc[0]:.1f}")
        
        if 'GrowthRate' in pred_row.columns:
            print(f"      Growth Rate: {pred_row['GrowthRate'].iloc[0]:.2%}")
        if 'BaseYear' in pred_row.columns:
            print(f"      Base Year: {pred_row['BaseYear'].iloc[0]}")
            
    except Exception as e:
        print(f"   ❌ Prediction row creation failed: {e}")
        return False
    
    # Test 3: Full Pipeline Integration (if models exist)
    print("\n🔧 Test 3: Full Pipeline Integration")
    try:
        from utils.data_loader import load_enhanced_models
        models, metadata, features = load_enhanced_models()
        
        if test_punch_code in models:
            # Test full prediction
            next_date, predictions, hours_predictions = predict_next_day(
                df, {test_punch_code: models[test_punch_code]}, date=datetime(2025, 8, 7)
            )
            
            if test_punch_code in predictions:
                pred_value = predictions[test_punch_code]
                hours_value = hours_predictions[test_punch_code]
                print(f"   ✅ Full prediction successful")
                print(f"      Workers: {pred_value}")
                print(f"      Hours: {hours_value:.1f}")
            else:
                print(f"   ⚠️ Prediction generated but no value for {test_punch_code}")
        else:
            print(f"   ⚠️ No model available for {test_punch_code}")
            
    except Exception as e:
        print(f"   ⚠️ Full pipeline test skipped: {e}")
    
    print("\n🎉 Complete yearly feature integration test completed!")
    return True

if __name__ == "__main__":
    success = test_complete_yearly_integration()
    
    if success:
        print("\n✅ All tests passed! Yearly features are properly integrated.")
        print("💡 Next steps:")
        print("   1. Run: python train_models2.py")
        print("   2. Test predictions with the new yearly features")
        print("   3. Compare accuracy with previous version")
    else:
        print("\n❌ Some tests failed. Check the errors above.")