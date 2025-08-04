"""
Robust Backtesting page for the Work Utilization Prediction app.
Enhanced statistical analysis and comprehensive model validation.
"""
import os
os.environ["STREAMLIT_SERVER_WATCH_CHANGES"] = "false"
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import sys
import traceback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import EnhancedFeatureTransformer
from utils.prediction import predict_multiple_days, evaluate_predictions
from utils.data_loader import load_combined_models, load_enhanced_models
from utils.sql_data_connector import extract_sql_data
from utils.holiday_utils import is_non_working_day, is_working_day_for_punch_code
from config import MODELS_DIR, DATA_DIR, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, ESSENTIAL_LAGS, ESSENTIAL_WINDOWS

# Configure page
st.set_page_config(
    page_title="Robust Model Backtesting",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'ts_data' not in st.session_state:
    st.session_state.ts_data = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

class StatisticalAnalyzer:
    """Advanced statistical analysis for backtesting results"""
    
    def __init__(self, results_df):
        self.results_df = results_df
        self.results_df['Error'] = self.results_df['Actual'] - self.results_df['Predicted']
        self.results_df['Absolute_Error'] = np.abs(self.results_df['Error'])
        self.results_df['Percentage_Error'] = np.where(
            self.results_df['Actual'] != 0,
            (self.results_df['Error'] / self.results_df['Actual']) * 100,
            0
        )
        self.results_df['Absolute_Percentage_Error'] = np.abs(self.results_df['Percentage_Error'])
    
    def calculate_comprehensive_metrics(self):
        """Calculate comprehensive statistical metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['MAE'] = mean_absolute_error(self.results_df['Actual'], self.results_df['Predicted'])
        metrics['RMSE'] = np.sqrt(mean_squared_error(self.results_df['Actual'], self.results_df['Predicted']))
        metrics['R2'] = r2_score(self.results_df['Actual'], self.results_df['Predicted'])
        
        # Advanced metrics
        metrics['MAPE'] = np.mean(self.results_df['Absolute_Percentage_Error'])
        metrics['Median_AE'] = np.median(self.results_df['Absolute_Error'])
        metrics['Mean_Error'] = np.mean(self.results_df['Error'])  # Bias
        metrics['Std_Error'] = np.std(self.results_df['Error'])
        
        # Distribution metrics
        metrics['Skewness'] = stats.skew(self.results_df['Error'])
        metrics['Kurtosis'] = stats.kurtosis(self.results_df['Error'])
        
        # Accuracy metrics
        threshold_5 = np.sum(self.results_df['Absolute_Percentage_Error'] <= 5) / len(self.results_df) * 100
        threshold_10 = np.sum(self.results_df['Absolute_Percentage_Error'] <= 10) / len(self.results_df) * 100
        threshold_20 = np.sum(self.results_df['Absolute_Percentage_Error'] <= 20) / len(self.results_df) * 100
        
        metrics['Accuracy_5%'] = threshold_5
        metrics['Accuracy_10%'] = threshold_10
        metrics['Accuracy_20%'] = threshold_20
        
        # Prediction interval metrics
        metrics['95%_CI_Lower'] = np.percentile(self.results_df['Error'], 2.5)
        metrics['95%_CI_Upper'] = np.percentile(self.results_df['Error'], 97.5)
        
        # Business impact metrics
        metrics['Total_Actual'] = self.results_df['Actual'].sum()
        metrics['Total_Predicted'] = self.results_df['Predicted'].sum()
        metrics['Total_Error'] = metrics['Total_Predicted'] - metrics['Total_Actual']
        metrics['Total_Error_Pct'] = (metrics['Total_Error'] / metrics['Total_Actual'] * 100) if metrics['Total_Actual'] > 0 else 0
        
        return metrics
    
    def analyze_by_dimension(self, dimension_col):
        """Analyze metrics by different dimensions"""
        dimension_metrics = []
        
        for dim_value in self.results_df[dimension_col].unique():
            dim_data = self.results_df[self.results_df[dimension_col] == dim_value]
            
            if len(dim_data) > 1:  # Need at least 2 points for meaningful metrics
                analyzer = StatisticalAnalyzer(dim_data)
                metrics = analyzer.calculate_comprehensive_metrics()
                metrics[dimension_col] = dim_value
                metrics['Sample_Size'] = len(dim_data)
                dimension_metrics.append(metrics)
        
        return pd.DataFrame(dimension_metrics)
    
    def detect_outliers(self, method='iqr'):
        """Detect outliers in prediction errors"""
        if method == 'iqr':
            Q1 = self.results_df['Absolute_Error'].quantile(0.25)
            Q3 = self.results_df['Absolute_Error'].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + 1.5 * IQR
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.results_df['Absolute_Error']))
            outlier_threshold = 3
            outliers = self.results_df[z_scores > outlier_threshold]
            return outliers
        
        outliers = self.results_df[self.results_df['Absolute_Error'] > outlier_threshold]
        return outliers
    
    def correlation_analysis(self):
        """Analyze correlations between errors and various factors"""
        correlations = {}
        
        # Numeric columns for correlation
        numeric_cols = ['Actual', 'Predicted', 'Error', 'Absolute_Error']
        
        # Add day features if available
        if 'Day of Week' in self.results_df.columns:
            # Convert day of week to numeric
            day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                          'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            self.results_df['Day_Numeric'] = self.results_df['Day of Week'].map(day_mapping)
            numeric_cols.append('Day_Numeric')
        
        # Calculate correlations
        corr_matrix = self.results_df[numeric_cols].corr()
        
        return corr_matrix

def load_workutilizationdata():
    """Load data from the WorkUtilizationData database table"""
    try:
        sql_query = """
        SELECT Date, PunchCode as WorkType, Hours, NoOfMan, SystemHours, Quantity, ResourceKPI, SystemKPI 
        FROM WorkUtilizationData
        WHERE PunchCode IN (215, 209, 213, 211, 214, 202, 203, 206, 208, 210, 217)
        ORDER BY Date
        """
        
        with st.spinner(f"Connecting to database {SQL_DATABASE} on {SQL_SERVER}..."):
            df = extract_sql_data(
                server=SQL_SERVER,
                database=SQL_DATABASE,
                query=sql_query,
                trusted_connection=SQL_TRUSTED_CONNECTION
            )
            
            if df is not None and not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                df['WorkType'] = df['WorkType'].astype(str)
                logger.info(f"Successfully loaded {len(df)} records from database")
                return df
            else:
                logger.warning("No data returned from database query")
                return None
    except Exception as e:
        logger.error(f"Error loading data from database: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def ensure_data_and_models():
    """Ensure data and models are loaded"""
    # Check if data is already loaded
    if st.session_state.df is None:
        st.session_state.df = load_workutilizationdata()
    
    if st.session_state.df is None:
        st.error("Could not load data from database. Please upload Excel file instead.")
        
        uploaded_file = st.file_uploader(
            "Upload Work Utilization Excel File", 
            type=["xlsx", "xls"],
            help="Upload Excel file with work utilization data"
        )
        
        use_sample_data = st.checkbox(
            "Use Sample Data", 
            value=False,
            help="Use sample data if you don't have your own file"
        )
        
        if uploaded_file is not None:
            st.session_state.df = load_data(uploaded_file)
            
        if use_sample_data:
            sample_path = os.path.join(DATA_DIR, "sample_work_utilization.xlsx")
            
            if os.path.exists(sample_path):
                st.session_state.df = load_data(sample_path)
        
        if st.session_state.df is None:
            st.warning("No data available. Please upload a file or connect to the database.")
            return False
    
    # Process data if available
    if st.session_state.df is not None and st.session_state.processed_df is None:
        with st.spinner("Processing data..."):
            # Initialize and use the transformer
            feature_transformer = EnhancedFeatureTransformer()
            
            # Fit and transform the data
            feature_transformer.fit(st.session_state.df)
            st.session_state.processed_df = feature_transformer.transform(st.session_state.df)
            st.session_state.ts_data = st.session_state.processed_df  # Already includes lag features
    
    # Check if we have models
    if st.session_state.models is None:
        with st.spinner("Loading models..."):
            # Try enhanced models first
            enhanced_models, enhanced_metadata, enhanced_features = load_enhanced_models()
            
            if enhanced_models:
                st.session_state.models = enhanced_models
                st.session_state.feature_importances = {}  # Enhanced models store importance differently
                st.session_state.metrics = {}  # Will be populated from metadata
                
                # Extract metrics from enhanced metadata
                for work_type, metadata in enhanced_metadata.items():
                    st.session_state.metrics[work_type] = {
                        'MAE': metadata.get('test_mae', 0),
                        'RMSE': metadata.get('test_rmse', 0),
                        'R²': metadata.get('test_r2', 0),
                        'MAPE': metadata.get('test_mape', 0)
                    }
                
                st.info("✅ Using Enhanced Pipeline Models for backtesting")
            # else:
            #     # Fall back to combined models
            #     models, feature_importances, metrics = load_combined_models()
                
            #     if models:
            #         st.session_state.models = models
            #         st.session_state.feature_importances = feature_importances
            #         st.session_state.metrics = metrics
            #         st.info("✅ Using Standard RandomForest Models for backtesting")
            #     else:
            #         st.error("No trained models available. Please run train_models2.py first.")
            #         return False
    
    return True

def run_comprehensive_backtesting(ts_data, models, backtest_days, work_types, model_type):
    """Run comprehensive backtesting with enhanced analysis"""
    
    try:
        # Determine neural network usage
        use_neural_backtest = model_type == "Neural Network"
        
        # Check neural network availability
        nn_available = False
        try:
            nn_path = os.path.join(MODELS_DIR, "work_utilization_nn_models.pkl")
            nn_available = os.path.exists(nn_path)
        except:
            pass
        
        if use_neural_backtest and not nn_available:
            st.warning("Neural network models are not available. Using Random Forest models for backtesting.")
            use_neural_backtest = False
        
        # Filter models to selected work types
        filtered_models = {wt: models[wt] for wt in work_types if wt in models}
        
        if not filtered_models:
            st.error("No models found for the selected punch codes.")
            return None
        
        # Get the data for backtesting
        max_date = ts_data['Date'].max()
        backtest_start = max_date - timedelta(days=backtest_days)
        
        # Check if we have enough data
        if ts_data['Date'].min() >= backtest_start:
            st.error(f"Not enough historical data for {backtest_days} days of backtesting. Please choose fewer days.")
            return None
        
        backtest_data = ts_data[ts_data['Date'] <= backtest_start]
        actual_data = ts_data[
            (ts_data['Date'] > backtest_start) & 
            (ts_data['Date'] <= max_date)
        ]
        
        # Generate predictions for the backtest period
        try:
            backtest_predictions, hours_predictions, holiday_info = predict_multiple_days(
                df=backtest_data,
                models=filtered_models,
                start_date=backtest_start,
                num_days=backtest_days,
                use_neural_network=use_neural_backtest
            )
        except Exception as pred_error:
            st.error(f"Error generating predictions: {str(pred_error)}")
            logger.error(f"Prediction error in backtesting: {str(pred_error)}")
            return None
        
        # Create comprehensive results dataframe
        results_records = []
        
        for date, predictions in backtest_predictions.items():
            # Check if day is non-working
            is_non_working, reason = is_non_working_day(date)
            
            for work_type, pred_value in predictions.items():
                if work_type in work_types:
                    # Apply non-working day logic
                    display_pred = 0 if is_non_working else pred_value
                    
                    # Get actual value
                    actual_records = actual_data[
                        (actual_data['Date'] == date) & 
                        (actual_data['WorkType'] == work_type)
                    ]
                    
                    actual_value = actual_records['NoOfMan'].sum() if not actual_records.empty else 0
                    
                    # Calculate hours prediction
                    hours_pred = 0
                    if date in hours_predictions and work_type in hours_predictions[date]:
                        hours_pred = hours_predictions[date][work_type]
                    
                    results_records.append({
                        'Date': date,
                        'Work Type': work_type,
                        'Predicted': display_pred,
                        'Predicted Hours': hours_pred,
                        'Actual': actual_value,
                        'Day of Week': date.strftime('%A'),
                        'Is Non-Working Day': "Yes" if is_non_working else "No",
                        'Reason': reason if is_non_working else "",
                        'Month': date.strftime('%B'),
                        'Week': date.isocalendar()[1],
                        'Day of Month': date.day
                    })
        
        results_df = pd.DataFrame(results_records)
        
        # Filter out records where actual is None
        results_df = results_df.dropna(subset=['Actual'])
        
        return results_df
        
    except Exception as e:
        st.error(f"Error during backtesting: {str(e)}")
        logger.error(f"Error during backtesting: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def display_comprehensive_metrics(analyzer):
    """Display comprehensive metrics in an organized way"""
    metrics = analyzer.calculate_comprehensive_metrics()
    
    # Organize metrics into categories
    st.subheader("📊 Comprehensive Performance Metrics")
    
    # Basic Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"{metrics['MAE']:.3f}")
        st.metric("RMSE", f"{metrics['RMSE']:.3f}")
    
    with col2:
        st.metric("R²", f"{metrics['R2']:.3f}")
        st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    
    with col3:
        st.metric("Median AE", f"{metrics['Median_AE']:.3f}")
        st.metric("Bias (Mean Error)", f"{metrics['Mean_Error']:.3f}")
    
    with col4:
        st.metric("Error Std Dev", f"{metrics['Std_Error']:.3f}")
        st.metric("Total Error %", f"{metrics['Total_Error_Pct']:.2f}%")
    
    # Advanced Statistical Metrics
    st.subheader("📈 Advanced Statistical Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Distribution Analysis**")
        st.write(f"Skewness: {metrics['Skewness']:.3f}")
        st.write(f"Kurtosis: {metrics['Kurtosis']:.3f}")
    
    with col2:
        st.write("**Prediction Accuracy**")
        st.write(f"Within 5%: {metrics['Accuracy_5%']:.1f}%")
        st.write(f"Within 10%: {metrics['Accuracy_10%']:.1f}%")
        st.write(f"Within 20%: {metrics['Accuracy_20%']:.1f}%")
    
    with col3:
        st.write("**Confidence Intervals**")
        st.write(f"95% CI Lower: {metrics['95%_CI_Lower']:.3f}")
        st.write(f"95% CI Upper: {metrics['95%_CI_Upper']:.3f}")
    
    # Business Impact
    st.subheader("💼 Business Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Actual Workers", f"{metrics['Total_Actual']:.0f}")
    
    with col2:
        st.metric("Total Predicted Workers", f"{metrics['Total_Predicted']:.0f}")
    
    with col3:
        delta_color = "inverse" if metrics['Total_Error'] > 0 else "normal"
        st.metric("Total Difference", f"{metrics['Total_Error']:.0f}", delta=f"{metrics['Total_Error_Pct']:.1f}%")

def create_advanced_visualizations(results_df, analyzer):
    """Create advanced visualizations for backtesting analysis"""
    
    # Error Distribution Analysis
    st.subheader("📊 Error Distribution Analysis")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Error Distribution', 'Prediction vs Actual', 
                       'Error Over Time', 'Error by Work Type'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Error histogram
    fig.add_trace(
        go.Histogram(x=results_df['Error'], name='Error Distribution', nbinsx=30),
        row=1, col=1
    )
    
    # Prediction vs Actual scatter
    fig.add_trace(
        go.Scatter(x=results_df['Actual'], y=results_df['Predicted'], 
                  mode='markers', name='Pred vs Actual',
                  hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'),
        row=1, col=2
    )
    
    # Perfect prediction line
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', name='Perfect Prediction', line=dict(dash='dash')),
        row=1, col=2
    )
    
    # Error over time
    daily_error = results_df.groupby('Date')['Error'].mean().reset_index()
    fig.add_trace(
        go.Scatter(x=daily_error['Date'], y=daily_error['Error'], 
                  mode='lines+markers', name='Daily Avg Error'),
        row=2, col=1
    )
    
    # Error by work type
    worktype_error = results_df.groupby('Work Type')['Absolute_Error'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=worktype_error['Work Type'], y=worktype_error['Absolute_Error'], 
               name='Avg Absolute Error'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Error Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("🔗 Correlation Analysis")
    
    corr_matrix = analyzer.correlation_analysis()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig_corr.update_layout(
        title="Feature Correlation Matrix",
        height=500
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

def analyze_temporal_patterns(results_df):
    """Analyze temporal patterns in prediction errors"""
    st.subheader("⏰ Temporal Pattern Analysis")
    
    # Day of week analysis
    dow_analysis = results_df.groupby('Day of Week').agg({
        'Absolute_Error': ['mean', 'std', 'count'],
        'Percentage_Error': 'mean',
        'Actual': 'mean',
        'Predicted': 'mean'
    }).round(3)
    
    dow_analysis.columns = ['Mean_AE', 'Std_AE', 'Count', 'Mean_PE', 'Mean_Actual', 'Mean_Predicted']
    dow_analysis = dow_analysis.reset_index()
    
    st.write("**Performance by Day of Week**")
    st.dataframe(dow_analysis, use_container_width=True)
    
    # Monthly analysis
    month_analysis = results_df.groupby('Month').agg({
        'Absolute_Error': ['mean', 'std', 'count'],
        'Percentage_Error': 'mean',
        'Actual': 'mean',
        'Predicted': 'mean'
    }).round(3)
    
    month_analysis.columns = ['Mean_AE', 'Std_AE', 'Count', 'Mean_PE', 'Mean_Actual', 'Mean_Predicted']
    month_analysis = month_analysis.reset_index()
    
    st.write("**Performance by Month**")
    st.dataframe(month_analysis, use_container_width=True)
    
    # Week-level trend
    weekly_trend = results_df.groupby('Week').agg({
        'Absolute_Error': 'mean',
        'Error': 'mean',
        'Actual': 'sum',
        'Predicted': 'sum'
    }).reset_index()
    
    fig_weekly = go.Figure()
    
    fig_weekly.add_trace(go.Scatter(
        x=weekly_trend['Week'],
        y=weekly_trend['Absolute_Error'],
        mode='lines+markers',
        name='Mean Absolute Error',
        yaxis='y'
    ))
    
    fig_weekly.add_trace(go.Scatter(
        x=weekly_trend['Week'],
        y=weekly_trend['Error'],
        mode='lines+markers',
        name='Mean Error (Bias)',
        yaxis='y2'
    ))
    
    fig_weekly.update_layout(
        title="Weekly Error Trends",
        xaxis_title="Week Number",
        yaxis=dict(title="Absolute Error", side="left"),
        yaxis2=dict(title="Error (Bias)", side="right", overlaying="y"),
        height=400
    )
    
    st.plotly_chart(fig_weekly, use_container_width=True)

def detect_and_analyze_outliers(analyzer, results_df):
    """Detect and analyze outliers in predictions"""
    st.subheader("🎯 Outlier Detection & Analysis")
    
    # Detect outliers using IQR method
    outliers_iqr = analyzer.detect_outliers(method='iqr')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Outliers (IQR)", len(outliers_iqr))
        st.metric("Outlier Percentage", f"{len(outliers_iqr)/len(results_df)*100:.2f}%")
    
    with col2:
        if len(outliers_iqr) > 0:
            st.metric("Max Outlier Error", f"{outliers_iqr['Absolute_Error'].max():.2f}")
            st.metric("Avg Outlier Error", f"{outliers_iqr['Absolute_Error'].mean():.2f}")
    
    # Display outliers
    if len(outliers_iqr) > 0:
        st.write("**Detected Outliers:**")
        outlier_display = outliers_iqr[['Date', 'Work Type', 'Actual', 'Predicted', 
                                       'Absolute_Error', 'Day of Week']].sort_values('Absolute_Error', ascending=False)
        st.dataframe(outlier_display, use_container_width=True)
        
        # Outlier patterns
        outlier_patterns = outliers_iqr.groupby('Work Type').size().reset_index(name='Outlier_Count')
        outlier_patterns = outlier_patterns.sort_values('Outlier_Count', ascending=False)
        
        fig_outliers = px.bar(outlier_patterns, x='Work Type', y='Outlier_Count',
                             title='Outliers by Work Type')
        st.plotly_chart(fig_outliers, use_container_width=True)

def main():
    st.header("🔍 Robust Model Backtesting & Statistical Analysis")
    
    st.write("""
    Comprehensive backtesting system with advanced statistical analysis for workforce prediction models.
    This system provides deep insights into model performance, error patterns, and business impact.
    """)
    
    # Check data and models
    if not ensure_data_and_models():
        return
    
    # Get available work types from models
    available_work_types = list(st.session_state.models.keys())
    
    # Backtesting Configuration
    st.subheader("⚙️ Backtesting Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Number of days for backtesting
        backtest_days = st.slider(
            "Backtesting Period (Days)",
            min_value=7,
            max_value=180,
            value=30,
            step=7,
            help="Number of days from the end of your dataset to use for validation"
        )
    
    with col2:
        # Model type selector
        model_type = st.radio(
            "Model Type",
            ["Random Forest", "Neural Network"],
            horizontal=True,
            help="Choose which model to use for backtesting"
        )
    
    with col3:
        # Work types selector
        selected_work_types = st.multiselect(
            "Punch Codes",
            options=available_work_types,
            default=available_work_types[:5] if len(available_work_types) > 5 else available_work_types,
            help="Select punch codes to include in backtesting"
        )
    
    # Run Backtesting
    if st.button("🚀 Run Comprehensive Backtesting", type="primary"):
        if not selected_work_types:
            st.warning("Please select at least one punch code for backtesting")
        else:
            with st.spinner(f"Running comprehensive backtesting for {backtest_days} days..."):
                
                # Run backtesting
                results_df = run_comprehensive_backtesting(
                    st.session_state.ts_data,
                    st.session_state.models,
                    backtest_days,
                    selected_work_types,
                    model_type
                )
                
                if results_df is not None and not results_df.empty:
                    # Initialize statistical analyzer
                    analyzer = StatisticalAnalyzer(results_df)
                    
                    # Display results overview
                    st.success(f"✅ Backtesting completed! Analyzed {len(results_df)} predictions.")
                    
                    # Create tabs for different analyses
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Overall Metrics", 
                        "📈 Visualizations", 
                        "⏰ Temporal Analysis",
                        "🎯 Outlier Analysis",
                        "📋 Detailed Results"
                    ])
                    
                    with tab1:
                        display_comprehensive_metrics(analyzer)
                        
                        # Work type specific analysis
                        st.subheader("📊 Performance by Work Type")
                        worktype_metrics = analyzer.analyze_by_dimension('Work Type')
                        if not worktype_metrics.empty:
                            display_cols = ['Work Type', 'Sample_Size', 'MAE', 'RMSE', 'R2', 'MAPE', 
                                          'Mean_Error', 'Accuracy_10%', 'Total_Error_Pct']
                            worktype_display = worktype_metrics[display_cols].round(3)
                            st.dataframe(worktype_display, use_container_width=True)
                    
                    with tab2:
                        create_advanced_visualizations(results_df, analyzer)
                    
                    with tab3:
                        analyze_temporal_patterns(results_df)
                    
                    with tab4:
                        detect_and_analyze_outliers(analyzer, results_df)
                    
                    with tab5:
                        st.subheader("📋 Detailed Backtesting Results")
                        
                        # Add filters for detailed view
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            filter_worktype = st.selectbox(
                                "Filter by Work Type",
                                options=['All'] + list(results_df['Work Type'].unique())
                            )
                        
                        with col2:
                            filter_day = st.selectbox(
                                "Filter by Day of Week",
                                options=['All'] + list(results_df['Day of Week'].unique())
                            )
                        
                        # Apply filters
                        filtered_results = results_df.copy()
                        if filter_worktype != 'All':
                            filtered_results = filtered_results[filtered_results['Work Type'] == filter_worktype]
                        if filter_day != 'All':
                            filtered_results = filtered_results[filtered_results['Day of Week'] == filter_day]
                        
                        # Display filtered results
                        display_columns = ['Date', 'Work Type', 'Actual', 'Predicted', 
                                         'Error', 'Absolute_Error', 'Percentage_Error', 'Day of Week']
                        
                        st.dataframe(
                            filtered_results[display_columns].round(3),
                            use_container_width=True,
                            column_config={
                                'Error': st.column_config.NumberColumn('Error', format="%.3f"),
                                'Absolute_Error': st.column_config.NumberColumn('Abs Error', format="%.3f"),
                                'Percentage_Error': st.column_config.NumberColumn('% Error', format="%.2f%%")
                            }
                        )
                        
                        # Download option
                        csv_data = filtered_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_data,
                            file_name=f"backtesting_results_{model_type.replace(' ', '_')}_{backtest_days}days.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.error("Failed to generate backtesting results")

if __name__ == "__main__":
    main()