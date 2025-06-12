"""
Feature Parameter Optimization Runner
Run this script to find optimal feature engineering parameters
"""

import pandas as pd
import logging
import sys
import os
from datetime import datetime
import json

# Add utils to path
sys.path.append('utils')

from utils.feature_optimization import run_feature_optimization
from utils.data_loader import load_data
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

def save_optimization_results(results, filename):
    """
    Save optimization results to file
    
    Parameters:
    -----------
    results : dict
        Optimization results
    filename : str
        Output filename
    """
    # Convert to JSON-serializable format
    serializable_results = {
        'best_config': results['best_config'],
        'best_performance': results['best_performance'],
        'summary': results['summary'],
        'timestamp': datetime.now().isoformat(),
        'total_combinations': len(results['all_results'])
    }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"📄 Results saved to {filename}")

def apply_best_config(best_config):
    """
    Update config.py with best parameters
    
    Parameters:
    -----------
    best_config : dict
        Best configuration found
    """
    logger.info("🔧 Updating config.py with optimal parameters...")
    
    # Read current config file
    with open('config.py', 'r') as f:
        config_content = f.read()
    
    # Create backup
    backup_filename = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    with open(backup_filename, 'w') as f:
        f.write(config_content)
    logger.info(f"📄 Backup saved as {backup_filename}")
    
    # Update configuration sections
    updates = []
    
    # Update FEATURE_GROUPS
    feature_groups_str = "FEATURE_GROUPS = {\n"
    for key, value in best_config['feature_groups'].items():
        feature_groups_str += f"    '{key}': {value},\n"
    feature_groups_str += "}"
    updates.append(('FEATURE_GROUPS = {', feature_groups_str))
    
    # Update ESSENTIAL_LAGS
    lags_str = f"ESSENTIAL_LAGS = {best_config['lag_days']}"
    updates.append(('ESSENTIAL_LAGS = ', lags_str))
    
    # Update ESSENTIAL_WINDOWS  
    windows_str = f"ESSENTIAL_WINDOWS = {best_config['rolling_windows']}"
    updates.append(('ESSENTIAL_WINDOWS = ', windows_str))
    
    # Update LAG_FEATURES_COLUMNS
    lag_cols_str = f"LAG_FEATURES_COLUMNS = {best_config['lag_columns']}"
    updates.append(('LAG_FEATURES_COLUMNS = ', lag_cols_str))
    
    # Update ROLLING_FEATURES_COLUMNS
    rolling_cols_str = f"ROLLING_FEATURES_COLUMNS = {best_config['rolling_columns']}"
    updates.append(('ROLLING_FEATURES_COLUMNS = ', rolling_cols_str))
    
    # Apply updates (simple approach - you may want to make this more robust)
    logger.info("⚠️  MANUAL UPDATE REQUIRED:")
    logger.info("Please update config.py with these optimal parameters:")
    logger.info("=" * 60)
    for search_text, replacement in updates:
        logger.info(f"{replacement}")
        logger.info("-" * 40)

def print_optimization_results(results):
    """
    Print formatted optimization results
    
    Parameters:
    -----------
    results : dict
        Optimization results
    """
    print("\n" + "=" * 80)
    print("🎯 FEATURE OPTIMIZATION RESULTS")
    print("=" * 80)
    
    best_perf = results['best_performance']
    print(f"✅ BEST PERFORMANCE:")
    print(f"   MAE: {best_perf['avg_mae']:.3f}")
    print(f"   R²:  {best_perf['avg_r2']:.3f}")
    print(f"   Valid Punch Codes: {best_perf['valid_punch_codes']}")
    
    print(f"\n📊 OPTIMIZATION SUMMARY:")
    summary = results['summary']
    print(f"   Combinations Tested: {summary['total_combinations_tested']}")
    print(f"   MAE Range: {summary['mae_range']['min']:.3f} - {summary['mae_range']['max']:.3f}")
    print(f"   Improvement Potential: {summary['improvement_potential']:.3f}")
    
    print(f"\n🔧 OPTIMAL CONFIGURATION:")
    best_config = results['best_config']
    
    print(f"   Feature Groups:")
    for key, value in best_config['feature_groups'].items():
        print(f"     {key}: {value}")
    
    print(f"   Lag Days: {best_config['lag_days']}")
    print(f"   Rolling Windows: {best_config['rolling_windows']}")
    print(f"   Lag Columns: {best_config['lag_columns']}")
    print(f"   Rolling Columns: {best_config['rolling_columns']}")
    
    print("\n" + "=" * 80)

def main():
    """
    Main optimization runner
    """
    logger.info("🚀 Starting Feature Parameter Optimization")
    logger.info(f"📊 Target Punch Codes: {config.ENHANCED_WORK_TYPES}")
    
    try:
        # Load data
        logger.info("📥 Loading training data...")
        df = load_data()
        
        if df is None or df.empty:
            logger.error("❌ No data available for optimization")
            return
            
        logger.info(f"✅ Loaded {len(df)} records")
        logger.info(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Run optimization
        logger.info("🔍 Running systematic parameter optimization...")
        results = run_feature_optimization(df, config.ENHANCED_WORK_TYPES)
        
        if results is None:
            logger.error("❌ Optimization failed")
            return
            
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"optimization_results_{timestamp}.json"
        save_optimization_results(results, results_file)
        
        # Print results
        print_optimization_results(results)
        
        # Suggest config updates
        apply_best_config(results['best_config'])
        
        logger.info("✅ Optimization completed successfully!")
        logger.info(f"📄 Detailed results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()