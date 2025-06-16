#!/usr/bin/env python3
"""
Quick script to check available functions in utils/feature_engineering.py
Run this to see what functions are available in your feature engineering module
"""

import sys
import inspect

def check_feature_engineering_functions():
    """Check what functions are available in feature_engineering.py"""
    
    try:
        import utils.feature_engineering as fe
        print("✅ Successfully imported utils.feature_engineering")
        print("=" * 60)
        
        # Get all functions (not starting with _)
        all_items = dir(fe)
        functions = []
        
        for item in all_items:
            if not item.startswith('_'):
                obj = getattr(fe, item)
                if callable(obj):
                    functions.append(item)
        
        print(f"📋 AVAILABLE FUNCTIONS ({len(functions)} found):")
        print("-" * 40)
        
        for func_name in sorted(functions):
            func = getattr(fe, func_name)
            try:
                # Get function signature
                sig = inspect.signature(func)
                print(f"  • {func_name}{sig}")
            except:
                print(f"  • {func_name}()")
        
        print("\n" + "=" * 60)
        print("💡 RECOMMENDED IMPORTS FOR best_feature_selector.py:")
        print("-" * 50)
        
        # Suggest imports based on function names
        recommended_functions = []
        
        # Check for lag features
        lag_functions = [f for f in functions if 'lag' in f.lower()]
        if lag_functions:
            print(f"📌 LAG FEATURES: {lag_functions}")
            recommended_functions.extend(lag_functions)
        
        # Check for rolling features  
        rolling_functions = [f for f in functions if 'roll' in f.lower()]
        if rolling_functions:
            print(f"📌 ROLLING FEATURES: {rolling_functions}")
            recommended_functions.extend(rolling_functions)
        
        # Check for date features
        date_functions = [f for f in functions if 'date' in f.lower()]
        if date_functions:
            print(f"📌 DATE FEATURES: {date_functions}")
            recommended_functions.extend(date_functions)
        
        # Check for cyclical features
        cyclical_functions = [f for f in functions if 'cyclic' in f.lower() or 'circular' in f.lower()]
        if cyclical_functions:
            print(f"📌 CYCLICAL FEATURES: {cyclical_functions}")
            recommended_functions.extend(cyclical_functions)
        
        # Check for trend features
        trend_functions = [f for f in functions if 'trend' in f.lower()]
        if trend_functions:
            print(f"📌 TREND FEATURES: {trend_functions}")
            recommended_functions.extend(trend_functions)
        
        # Check for pattern features
        pattern_functions = [f for f in functions if 'pattern' in f.lower()]
        if pattern_functions:
            print(f"📌 PATTERN FEATURES: {pattern_functions}")
            recommended_functions.extend(pattern_functions)
        
        if recommended_functions:
            print(f"\n🔧 SUGGESTED IMPORT STATEMENT:")
            print("-" * 30)
            print("from utils.feature_engineering import (")
            for func in recommended_functions:
                print(f"    {func},")
            print(")")
        
        print(f"\n✅ Analysis complete! Found {len(functions)} callable functions.")
        
        return functions
        
    except ImportError as e:
        print(f"❌ Error importing utils.feature_engineering: {e}")
        print("\n🔍 TROUBLESHOOTING:")
        print("1. Make sure you're in the correct directory")
        print("2. Check if utils/feature_engineering.py exists")
        print("3. Check if utils/__init__.py exists")
        return None
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def check_train_models2():
    """Check if train_models2.py exists and what it contains"""
    import os
    
    print("\n" + "=" * 60)
    print("🔍 CHECKING train_models2.py:")
    print("-" * 30)
    
    if os.path.exists("train_models2.py"):
        print("✅ train_models2.py found")
        
        # Look for EnhancedFeatureTransformer
        try:
            with open("train_models2.py", 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "EnhancedFeatureTransformer" in content:
                print("✅ EnhancedFeatureTransformer found in train_models2.py")
            else:
                print("⚠️  EnhancedFeatureTransformer not found in train_models2.py")
                
            if "create_enhanced_features" in content:
                print("✅ create_enhanced_features method found")
            else:
                print("⚠️  create_enhanced_features method not found")
                
            if "Pipeline" in content:
                print("✅ Pipeline usage found")
            else:
                print("⚠️  Pipeline usage not found")
                
        except Exception as e:
            print(f"❌ Error reading train_models2.py: {e}")
    else:
        print("❌ train_models2.py not found")

def main():
    print("🔍 FEATURE ENGINEERING FUNCTION CHECKER")
    print("=" * 60)
    print("This script helps identify available functions in your feature engineering module")
    print()
    
    # Check feature engineering functions
    functions = check_feature_engineering_functions()
    
    # Check train_models2.py
    check_train_models2()
    
    print(f"\n{'='*60}")
    print("📝 NEXT STEPS:")
    print("-" * 15)
    
    if functions:
        print("1. ✅ Use the suggested import statement in best_feature_selector.py")
        print("2. ✅ Update the function calls in EnhancedFeatureTransformer")
        print("3. ✅ Run the feature selection: python best_feature_selector.py")
    else:
        print("1. ❌ Fix the import issues first")
        print("2. ❌ Make sure utils/feature_engineering.py is accessible")
        print("3. ❌ Check your Python path and working directory")
    
    print("4. 📧 Share the output of this script if you need further help")

if __name__ == "__main__":
    main()