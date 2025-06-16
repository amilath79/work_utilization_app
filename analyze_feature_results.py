import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

class FeatureResultsAnalyzer:
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self.load_results()
        self.df = self.create_results_dataframe()
    
    def load_results(self):
        """Load results from JSON file"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def create_results_dataframe(self):
        """Convert results to DataFrame for analysis"""
        rows = []
        
        for result in self.results:
            config = result['config']
            perf = result['performance']
            
            row = {
                'config_id': result['config_id'],
                'mae_mean': perf['mae_mean'],
                'mae_std': perf['mae_std'],
                'r2_mean': perf['r2_mean'],
                'r2_std': perf['r2_std'],
                'mape_mean': perf['mape_mean'],
                'feature_count': perf['feature_count'],
                'training_time': result['training_time'],
            }
            
            # Add feature flags
            feature_types = ['LAG_FEATURES', 'ROLLING_FEATURES', 'DATE_FEATURES', 
                           'CYCLICAL_FEATURES', 'TREND_FEATURES', 'PATTERN_FEATURES']
            for ft in feature_types:
                row[ft] = config.get(ft, False)
            
            # Add parameter info
            if 'lag_params' in config:
                row['lag_count'] = len(config['lag_params']['lags'])
                row['max_lag'] = max(config['lag_params']['lags'])
            else:
                row['lag_count'] = 0
                row['max_lag'] = 0
                
            if 'rolling_params' in config:
                row['window_count'] = len(config['rolling_params']['windows'])
                row['max_window'] = max(config['rolling_params']['windows'])
            else:
                row['window_count'] = 0
                row['max_window'] = 0
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_summary_report(self):
        """Generate summary statistics"""
        print("=" * 80)
        print("FEATURE SELECTION SUMMARY REPORT")
        print("=" * 80)
        
        print(f"\nTotal Configurations Tested: {len(self.df)}")
        print(f"Best MAE: {self.df['mae_mean'].min():.4f}")
        print(f"Best R²: {self.df['r2_mean'].max():.4f}")
        print(f"Best MAPE: {self.df['mape_mean'].min():.2f}%")
        
        # Performance distribution
        print("\nPerformance Distribution:")
        print(f"MAE - Mean: {self.df['mae_mean'].mean():.4f}, Std: {self.df['mae_mean'].std():.4f}")
        print(f"R² - Mean: {self.df['r2_mean'].mean():.4f}, Std: {self.df['r2_mean'].std():.4f}")
        print(f"MAPE - Mean: {self.df['mape_mean'].mean():.2f}%, Std: {self.df['mape_mean'].std():.2f}%")
        
        # Top 10 configurations
        print("\n🏆 TOP 10 CONFIGURATIONS BY MAE:")
        print("-" * 50)
        top_10 = self.df.nsmallest(10, 'mae_mean')
        
        for idx, row in top_10.iterrows():
            active_features = [ft for ft in ['LAG_FEATURES', 'ROLLING_FEATURES', 'DATE_FEATURES', 
                                           'CYCLICAL_FEATURES', 'TREND_FEATURES', 'PATTERN_FEATURES'] 
                             if row[ft]]
            
            print(f"#{row.name + 1:2d} | MAE: {row['mae_mean']:.4f} | R²: {row['r2_mean']:.4f} | "
                  f"Features: {len(active_features)} | {', '.join(active_features)}")
    
    def analyze_feature_impact(self):
        """Analyze impact of each feature type"""
        print("\n📊 FEATURE IMPACT ANALYSIS:")
        print("-" * 40)
        
        feature_types = ['LAG_FEATURES', 'ROLLING_FEATURES', 'DATE_FEATURES', 
                        'CYCLICAL_FEATURES', 'TREND_FEATURES', 'PATTERN_FEATURES']
        
        for feature in feature_types:
            with_feature = self.df[self.df[feature] == True]['mae_mean']
            without_feature = self.df[self.df[feature] == False]['mae_mean']
            
            if len(with_feature) > 0 and len(without_feature) > 0:
                improvement = ((without_feature.mean() - with_feature.mean()) / without_feature.mean()) * 100
                p_value = self.statistical_test(with_feature, without_feature)
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                print(f"{feature:20} | Impact: {improvement:+6.2f}% | "
                      f"Avg MAE: {with_feature.mean():.4f} vs {without_feature.mean():.4f} {significance}")
    
    def statistical_test(self, group1, group2):
        """Simple t-test for feature impact"""
        from scipy import stats
        try:
            _, p_value = stats.ttest_ind(group1, group2)
            return p_value
        except:
            return 1.0
    
    def analyze_parameter_optimization(self):
        """Analyze optimal parameters"""
        print("\n⚙️ PARAMETER OPTIMIZATION ANALYSIS:")
        print("-" * 45)
        
        # Lag parameter analysis
        lag_configs = self.df[self.df['LAG_FEATURES'] == True]
        if len(lag_configs) > 0:
            print("\nLAG PARAMETER ANALYSIS:")
            lag_analysis = lag_configs.groupby('lag_count')['mae_mean'].agg(['mean', 'std', 'count'])
            print(lag_analysis.round(4))
            
            best_lag_count = lag_analysis['mean'].idxmin()
            print(f"Optimal lag count: {best_lag_count}")
        
        # Rolling window analysis
        rolling_configs = self.df[self.df['ROLLING_FEATURES'] == True]
        if len(rolling_configs) > 0:
            print("\nROLLING WINDOW ANALYSIS:")
            window_analysis = rolling_configs.groupby('window_count')['mae_mean'].agg(['mean', 'std', 'count'])
            print(window_analysis.round(4))
            
            best_window_count = window_analysis['mean'].idxmin()
            print(f"Optimal window count: {best_window_count}")
    
    def complexity_vs_performance(self):
        """Analyze feature count vs performance"""
        print("\n🎯 COMPLEXITY vs PERFORMANCE ANALYSIS:")
        print("-" * 45)
        
        complexity_analysis = self.df.groupby('feature_count')['mae_mean'].agg(['mean', 'std', 'count'])
        print("Feature Count | Avg MAE  | Std     | Count")
        print("-" * 40)
        for features, row in complexity_analysis.iterrows():
            print(f"{features:12d} | {row['mean']:8.4f} | {row['std']:7.4f} | {row['count']:5.0f}")
        
        # Find optimal complexity
        optimal_features = complexity_analysis['mean'].idxmin()
        print(f"\nOptimal feature count: {optimal_features}")
        
        # Efficiency analysis (performance per feature)
        self.df['efficiency'] = 1 / (self.df['mae_mean'] * self.df['feature_count'])
        most_efficient = self.df.loc[self.df['efficiency'].idxmax()]
        print(f"Most efficient config: {most_efficient['config_id']} "
              f"(MAE: {most_efficient['mae_mean']:.4f}, Features: {most_efficient['feature_count']})")
    
    def generate_plots(self, save_plots=True):
        """Generate visualization plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: MAE distribution
        axes[0, 0].hist(self.df['mae_mean'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(self.df['mae_mean'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].set_xlabel('MAE')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('MAE Distribution')
        axes[0, 0].legend()
        
        # Plot 2: Feature count vs MAE
        axes[0, 1].scatter(self.df['feature_count'], self.df['mae_mean'], alpha=0.6)
        axes[0, 1].set_xlabel('Feature Count')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Feature Count vs MAE')
        
        # Plot 3: Feature impact
        feature_types = ['LAG_FEATURES', 'ROLLING_FEATURES', 'DATE_FEATURES', 
                        'CYCLICAL_FEATURES', 'TREND_FEATURES', 'PATTERN_FEATURES']
        impacts = []
        for feature in feature_types:
            with_feature = self.df[self.df[feature] == True]['mae_mean'].mean()
            without_feature = self.df[self.df[feature] == False]['mae_mean'].mean()
            if not np.isnan(with_feature) and not np.isnan(without_feature):
                impact = ((without_feature - with_feature) / without_feature) * 100
                impacts.append(impact)
            else:
                impacts.append(0)
        
        colors = ['green' if x > 0 else 'red' for x in impacts]
        axes[1, 0].bar(range(len(feature_types)), impacts, color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Feature Type')
        axes[1, 0].set_ylabel('MAE Improvement (%)')
        axes[1, 0].set_title('Feature Impact on MAE')
        axes[1, 0].set_xticks(range(len(feature_types)))
        axes[1, 0].set_xticklabels([ft.replace('_FEATURES', '') for ft in feature_types], rotation=45)
        
        # Plot 4: MAE vs R²
        axes[1, 1].scatter(self.df['mae_mean'], self.df['r2_mean'], alpha=0.6)
        axes[1, 1].set_xlabel('MAE')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('MAE vs R² Trade-off')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('feature_selection_analysis.png', dpi=300, bbox_inches='tight')
            print("\nPlots saved as 'feature_selection_analysis.png'")
        
        plt.show()
    
    def export_best_config(self, output_file='best_config.py'):
        """Export best configuration as Python config file"""
        best_config = self.df.loc[self.df['mae_mean'].idxmin()]
        
        config_text = f"""# Best Feature Configuration
# Generated from feature selection analysis
# Expected MAE: {best_config['mae_mean']:.4f}
# Expected R²: {best_config['r2_mean']:.4f}
# Expected MAPE: {best_config['mape_mean']:.2f}%

# Feature Flags
LAG_FEATURES = {best_config['LAG_FEATURES']}
ROLLING_FEATURES = {best_config['ROLLING_FEATURES']}
DATE_FEATURES = {best_config['DATE_FEATURES']}
CYCLICAL_FEATURES = {best_config['CYCLICAL_FEATURES']}
TREND_FEATURES = {best_config['TREND_FEATURES']}
PATTERN_FEATURES = {best_config['PATTERN_FEATURES']}

# Parameters (update these in your feature engineering)
"""
        
        # Get original config for parameters
        original_config = self.results[int(best_config['config_id']) - 1]['config']
        if 'lag_params' in original_config:
            config_text += f"LAG_PERIODS = {original_config['lag_params']['lags']}\n"
        if 'rolling_params' in original_config:
            config_text += f"ROLLING_WINDOWS = {original_config['rolling_params']['windows']}\n"
        
        with open(output_file, 'w') as f:
            f.write(config_text)
        
        print(f"\nBest configuration exported to '{output_file}'")
    
    def run_full_analysis(self, generate_plots=True, export_config=True):
        """Run complete analysis pipeline"""
        self.generate_summary_report()
        self.analyze_feature_impact()
        self.analyze_parameter_optimization()
        self.complexity_vs_performance()
        
        if generate_plots:
            try:
                self.generate_plots()
            except Exception as e:
                print(f"Could not generate plots: {e}")
        
        if export_config:
            self.export_best_config()


def main():
    parser = argparse.ArgumentParser(description='Analyze feature selection results')
    parser.add_argument('results_file', help='Path to feature selection results JSON file')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--no-export', action='store_true', help='Skip config export')
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: Results file '{args.results_file}' not found")
        return
    
    analyzer = FeatureResultsAnalyzer(args.results_file)
    analyzer.run_full_analysis(
        generate_plots=not args.no_plots,
        export_config=not args.no_export
    )


if __name__ == "__main__":
    # If no arguments provided, try to find the most recent results file
    import sys
    if len(sys.argv) == 1:
        results_files = list(Path('.').glob('feature_selection_results_*.json'))
        if results_files:
            latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
            print(f"Using latest results file: {latest_file}")
            analyzer = FeatureResultsAnalyzer(str(latest_file))
            analyzer.run_full_analysis()
        else:
            print("No results files found. Please run feature selection first.")
    else:
        main()