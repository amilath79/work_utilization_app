"""
Backtesting page for the Work Utilization Prediction app.
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

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import engineer_features, create_lag_features
from utils.prediction import predict_multiple_days, evaluate_predictions
from utils.data_loader import load_combined_models, load_data
from utils.sql_data_connector import extract_sql_data
from utils.holiday_utils import is_non_working_day
from config import MODELS_DIR, DATA_DIR, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, LAG_DAYS, ROLLING_WINDOWS

# Configure page
st.set_page_config(
    page_title="Model Backtesting",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

# Create session state for data persistence if not present
if 'full_df' not in st.session_state:
    st.session_state.full_df = None
if 'full_processed_df' not in st.session_state:
    st.session_state.full_processed_df = None
if 'full_ts_data' not in st.session_state:
    st.session_state.full_ts_data = None

if 'basic_df' not in st.session_state:
    st.session_state.basic_df = None
if 'basic_processed_df' not in st.session_state:
    st.session_state.basic_processed_df = None
if 'basic_ts_data' not in st.session_state:
    st.session_state.basic_ts_data = None

if 'models' not in st.session_state:
    st.session_state.models = None
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

def load_workutilizationdata():
    """
    Load data from the WorkUtilizationData table
    """
    try:
        # Create SQL query focused on full-feature dataset
        sql_query = """
        SELECT Date, PunchCode as WorkType, Hours, NoOfMan, SystemHours, Quantity, ResourceKPI, SystemKPI 
        FROM WorkUtilizationData
        WHERE PunchCode IN (215, 209, 213, 211, 214)
        ORDER BY Date
        """
        
        # Show connecting message
        with st.spinner(f"Connecting to database {SQL_DATABASE} on {SQL_SERVER} for WorkUtilizationData..."):
            # Use the extract_sql_data function
            df = extract_sql_data(
                server=SQL_SERVER,
                database=SQL_DATABASE,
                query=sql_query,
                trusted_connection=SQL_TRUSTED_CONNECTION
            )
            
            if df is not None and not df.empty:
                # Ensure Date is datetime type
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Ensure WorkType is string
                df['WorkType'] = df['WorkType'].astype(str)
                
                logger.info(f"Successfully loaded {len(df)} records from WorkUtilizationData")
                return df
            else:
                logger.warning("No data returned from WorkUtilizationData query")
                return None
    except Exception as e:
        logger.error(f"Error loading data from WorkUtilizationData: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_workdata1():
    """
    Load data from the WorkData1 table
    """
    try:
        # Create SQL query focused on basic dataset
        sql_query = """
        SELECT Date, PunchCode as WorkType, Hours, NoOfMan
        FROM WorkData1
        WHERE PunchCode IN (202, 203, 206, 208, 210, 217)
        ORDER BY Date
        """
        
        # Show connecting message
        with st.spinner(f"Connecting to database {SQL_DATABASE} on {SQL_SERVER} for WorkData1..."):
            # Use the extract_sql_data function
            df = extract_sql_data(
                server=SQL_SERVER,
                database=SQL_DATABASE,
                query=sql_query,
                trusted_connection=SQL_TRUSTED_CONNECTION
            )
            
            if df is not None and not df.empty:
                # Ensure Date is datetime type
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Ensure WorkType is string
                df['WorkType'] = df['WorkType'].astype(str)
                
                logger.info(f"Successfully loaded {len(df)} records from WorkData1")
                return df
            else:
                logger.warning("No data returned from WorkData1 query")
                return None
    except Exception as e:
        logger.error(f"Error loading data from WorkData1: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Check if we have data and models
def ensure_data_and_models():
    # Check if we have data - try to load from database if not loaded yet
    if st.session_state.full_df is None:
        st.session_state.full_df = load_workutilizationdata()
    
    if st.session_state.basic_df is None:
        st.session_state.basic_df = load_workdata1()
    
    # If database load failed, offer Excel options
    if st.session_state.full_df is None and st.session_state.basic_df is None:
        st.error("Could not load data from database. Please upload Excel files instead.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Full Features Dataset")
            uploaded_full = st.file_uploader(
                "Upload Full Features Excel File", 
                type=["xlsx", "xls"],
                key="full_uploader",
                help="Upload Excel file with full features"
            )
            
            if uploaded_full is not None:
                st.session_state.full_df = load_data(uploaded_full)
        
        with col2:
            st.subheader("Basic Dataset")
            uploaded_basic = st.file_uploader(
                "Upload Basic Excel File", 
                type=["xlsx", "xls"],
                key="basic_uploader",
                help="Upload Excel file with basic features"
            )
            
            if uploaded_basic is not None:
                st.session_state.basic_df = load_data(uploaded_basic)
        
        use_sample_data = st.checkbox(
            "Use Sample Data", 
            value=False,
            help="Use sample data if you don't have your own files"
        )
        
        if use_sample_data:
            sample_full_path = os.path.join(DATA_DIR, "sample_full_features.xlsx")
            sample_basic_path = os.path.join(DATA_DIR, "sample_basic_features.xlsx")
            
            if os.path.exists(sample_full_path):
                st.session_state.full_df = load_data(sample_full_path)
            
            if os.path.exists(sample_basic_path):
                st.session_state.basic_df = load_data(sample_basic_path)
        
        # Check if we have at least one dataset after trying upload options
        if st.session_state.full_df is None and st.session_state.basic_df is None:
            st.warning("No data available. Please upload files or connect to the database.")
            return False
    
    # Process full data if available
    if st.session_state.full_df is not None and st.session_state.full_processed_df is None:
        with st.spinner("Processing full features data..."):
            st.session_state.full_processed_df = engineer_features(st.session_state.full_df)
            # Use full lag days configuration for full features
            full_lag_days = LAG_DAYS
            full_rolling_windows = ROLLING_WINDOWS
            st.session_state.full_ts_data = create_lag_features(
                st.session_state.full_processed_df,
                lag_days=full_lag_days,
                rolling_windows=full_rolling_windows
            )
    
    # Process basic data if available
    if st.session_state.basic_df is not None and st.session_state.basic_processed_df is None:
        with st.spinner("Processing basic data..."):
            st.session_state.basic_processed_df = engineer_features(st.session_state.basic_df)
            # Use limited lag days configuration for basic features
            basic_lag_days = [1, 2, 3, 7]  # Limited lag days for basic features
            basic_rolling_windows = [7]     # Limited rolling windows for basic features
            st.session_state.basic_ts_data = create_lag_features(
                st.session_state.basic_processed_df,
                lag_days=basic_lag_days,
                rolling_windows=basic_rolling_windows
            )
    
    # Check if we have models
    if st.session_state.models is None:
        with st.spinner("Loading models..."):
            models, feature_importances, metrics = load_combined_models()
            
            if models:
                st.session_state.models = models
                st.session_state.feature_importances = feature_importances
                st.session_state.metrics = metrics
            else:
                st.error("No trained models available. Please train models first.")
                return False
    
    return True

def main():
    st.header("Model Backtesting & Validation")
    
    st.write("""
    Backtesting allows you to evaluate how well the models perform by comparing
    predictions against actual historical data.
    """)
    
    # Check data and models
    if not ensure_data_and_models():
        return
    
    # Create tabs for different datasets
    dataset_tabs = st.tabs([
        "Full Features Dataset (PunchCodes: 215, 209, 213, 211, 214)", 
        "Basic Dataset (PunchCodes: 202, 203, 206, 208, 210, 217)"
    ])
    
    # Full Features Dataset Tab
    with dataset_tabs[0]:
        if st.session_state.full_df is not None and st.session_state.full_ts_data is not None:
            # Verify the data structure before passing it to run_backtesting
            if isinstance(st.session_state.full_ts_data, pd.DataFrame) and 'Date' in st.session_state.full_ts_data.columns:
                run_backtesting(
                    dataset_name="Full Features",
                    ts_data=st.session_state.full_ts_data,
                    punch_codes=["215", "209", "213", "211", "214"]
                )
            else:
                st.error("Full features dataset is not properly structured. Missing 'Date' column or not a DataFrame.")
        else:
            st.warning("Full features dataset not available.")
    
    # Basic Dataset Tab
    with dataset_tabs[1]:
        if st.session_state.basic_df is not None and st.session_state.basic_ts_data is not None:
            # Verify the data structure before passing it to run_backtesting
            if isinstance(st.session_state.basic_ts_data, pd.DataFrame) and 'Date' in st.session_state.basic_ts_data.columns:
                run_backtesting(
                    dataset_name="Basic",
                    ts_data=st.session_state.basic_ts_data,
                    punch_codes=["202", "203", "206", "208", "210", "217"]
                )
            else:
                st.error("Basic dataset is not properly structured. Missing 'Date' column or not a DataFrame.")
        else:
            st.warning("Basic dataset not available.")

def run_backtesting(dataset_name, ts_data, punch_codes):
    """Run backtesting for the specified dataset"""
    st.subheader(f"{dataset_name} Dataset Backtesting")
    
    # Additional safety check for the ts_data
    if ts_data is None or not isinstance(ts_data, pd.DataFrame) or 'Date' not in ts_data.columns:
        st.error(f"Invalid data for {dataset_name} dataset. Cannot perform backtesting.")
        return
    
    # Get all work types from models that match our punch codes
    available_work_types = [wt for wt in st.session_state.models.keys() if wt in punch_codes]
    
    if not available_work_types:
        st.warning(f"No models found for the punch codes in the {dataset_name} dataset.")
        st.info(f"Available models: {sorted(list(st.session_state.models.keys()))}")
        return
    
    # Backtesting options
    st.write("### Backtesting Options")
    
    # Number of days for backtesting
    backtest_days = st.slider(
        "Number of Days for Backtesting",
        min_value=7,
        max_value=90,
        value=30,
        step=7,
        help="Select number of days from the end of your dataset to use for validation",
        key=f"{dataset_name}_backtest_days"
    )
    
    # Work types selector
    backtest_work_types = st.multiselect(
        "Select Punch Codes for Backtesting",
        options=available_work_types,
        default=available_work_types[:3] if len(available_work_types) > 3 else available_work_types,
        help="Select Punch Codes to include in backtesting",
        key=f"{dataset_name}_backtest_work_types"
    )
    
    # Model type selector
    backtest_model_type = st.radio(
        "Select Model Type for Backtesting",
        ["Random Forest", "Neural Network"],
        horizontal=True,
        help="Choose which model to use for backtesting",
        key=f"{dataset_name}_backtest_model"
    )
    
    use_neural_backtest = backtest_model_type == "Neural Network"
    
    # Check if neural network models are available
    nn_available = False
    try:
        nn_path = os.path.join(MODELS_DIR, "work_utilization_nn_models.pkl")
        nn_available = os.path.exists(nn_path)
    except:
        pass
    
    if use_neural_backtest and not nn_available:
        st.warning("Neural network models are not available. Using Random Forest models for backtesting.")
        use_neural_backtest = False
    
    # Run backtesting button
    if st.button("Run Backtesting", type="primary", key=f"{dataset_name}_backtest_button"):
        if not backtest_work_types:
            st.warning("Please select at least one Punch Code for backtesting")
        else:
            with st.spinner(f"Running backtesting for the last {backtest_days} days..."):
                try:
                    # Filter models to selected work types
                    filtered_models = {wt: st.session_state.models[wt] for wt in backtest_work_types if wt in st.session_state.models}
                    
                    if not filtered_models:
                        st.error("No models found for the selected punch codes.")
                        return
                    
                    # Get the data for backtesting
                    max_date = ts_data['Date'].max()
                    backtest_start = max_date - timedelta(days=backtest_days)
                    
                    # Check if we have enough data
                    if ts_data['Date'].min() >= backtest_start:
                        st.error(f"Not enough historical data for {backtest_days} days of backtesting. Please choose fewer days.")
                        return
                    
                    backtest_data = ts_data[ts_data['Date'] <= backtest_start]
                    actual_data = ts_data[
                        (ts_data['Date'] > backtest_start) & 
                        (ts_data['Date'] <= max_date)
                    ]
                    
                    # Generate predictions for the backtest period
                    # KEY FIX: Now unpacking THREE return values - predictions, hours, and holiday info
                    backtest_predictions, hours_predictions, holiday_info = predict_multiple_days(
                        df=backtest_data,
                        models=filtered_models,
                        num_days=backtest_days,
                        use_neural_network=use_neural_backtest
                    )
                    
                    # Create a dataframe with predictions and actuals
                    results_records = []
                    metrics_by_worktype = {}
                    
                    for date, predictions in backtest_predictions.items():
                        # Explicitly check if day is non-working
                        is_non_working, reason = is_non_working_day(date)
                        
                        for work_type, pred_value in predictions.items():
                            if work_type in backtest_work_types:
                                # Zero for non-working days except Sunday
                                display_pred = 0 if is_non_working else pred_value
                                
                                # Get actual value
                                actual_records = actual_data[
                                    (actual_data['Date'] == date) & 
                                    (actual_data['WorkType'] == work_type)
                                ]
                                
                                actual_value = actual_records['NoOfMan'].sum() if not actual_records.empty else None
                                
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
                                    'Difference': actual_value - display_pred if actual_value is not None else None,
                                    'Day of Week': date.strftime('%A'),
                                    'Is Non-Working Day': "Yes" if is_non_working else "No",
                                    'Reason': reason if is_non_working else ""
                                })
                    
                    results_df = pd.DataFrame(results_records)
                    
                    # Calculate metrics by work type
                    for work_type in backtest_work_types:
                        wt_results = results_df[results_df['Work Type'] == work_type].dropna(subset=['Actual'])
                        
                        if len(wt_results) > 0:
                            metrics = evaluate_predictions(
                                wt_results['Actual'].values,
                                wt_results['Predicted'].values
                            )
                            metrics_by_worktype[work_type] = metrics
                    
                    # Display results
                    st.subheader(f"Backtesting Results using {backtest_model_type}")
                    
                    # Display metrics
                    st.write("### Model Performance Metrics")
                    
                    metrics_records = []
                    for work_type, metrics in metrics_by_worktype.items():
                        metrics_records.append({
                            'Work Type': work_type,
                            'MAE': metrics['MAE'],
                            'RMSE': metrics['RMSE'],
                            'R²': metrics['R²'],
                            'MAPE (%)': metrics['MAPE']
                        })
                    
                    metrics_df = pd.DataFrame(metrics_records)
                    
                    st.dataframe(
                        metrics_df,
                        use_container_width=True,
                        column_config={
                            'MAE': st.column_config.NumberColumn('MAE', format="%.4f"),
                            'RMSE': st.column_config.NumberColumn('RMSE', format="%.4f"),
                            'R²': st.column_config.NumberColumn('R²', format="%.4f"),
                            'MAPE (%)': st.column_config.NumberColumn('MAPE (%)', format="%.2f")
                        }
                    )
                    
                    if not metrics_records:
                        st.warning("No metrics could be calculated. This could be due to missing actual data for comparison.")
                        return
                    
                    # Create summary chart
                    st.write("### Performance Comparison")
                    
                    fig = go.Figure()
                    
                    # Add MAE bars
                    fig.add_trace(go.Bar(
                        x=metrics_df['Work Type'],
                        y=metrics_df['MAE'],
                        name="MAE",
                        marker_color="blue"
                    ))
                    
                    # Add R² line on secondary axis
                    fig.add_trace(go.Scatter(
                        x=metrics_df['Work Type'],
                        y=metrics_df['R²'],
                        name="R²",
                        mode="lines+markers",
                        marker=dict(size=8, color="green"),
                        line=dict(width=2, dash="dot"),
                        yaxis="y2"
                    ))
                    
                    # Update layout with second y-axis
                    fig.update_layout(
                        title="Model Performance by Work Type",
                        xaxis_title="Work Type",
                        yaxis_title="Mean Absolute Error (MAE)",
                        yaxis2=dict(
                            title="R² Score",
                            overlaying="y",
                            side="right",
                            range=[0, 1]
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Actual vs Predicted comparison
                    st.subheader("Actual vs Predicted Comparison")
                    
                    # Work type selector for the chart
                    selected_work_type_for_chart = st.selectbox(
                        "Select a Punch Code for Chart",
                        options=backtest_work_types,
                        index=0 if backtest_work_types else None,
                        key=f"{dataset_name}_chart_work_type"
                    )
                    
                    # Create and display the chart
                    if selected_work_type_for_chart:
                        chart_data = results_df[results_df['Work Type'] == selected_work_type_for_chart].copy()
                        
                        # Filter out rows with missing actual values
                        chart_data = chart_data.dropna(subset=['Actual'])
                        
                        if len(chart_data) > 0:
                            # Create figure
                            fig = go.Figure()
                            
                            # Add actual line
                            fig.add_trace(go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['Actual'],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue', width=2),
                                marker=dict(size=8)
                            ))
                            
                            # Add predicted line
                            fig.add_trace(go.Scatter(
                                x=chart_data['Date'],
                                y=chart_data['Predicted'],
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=8)
                            ))
                            
                            # Update layout
                            title = f"Actual vs Predicted Values for {selected_work_type_for_chart}"
                            
                            fig.update_layout(
                                title=title,
                                xaxis_title='Date',
                                yaxis_title='Number of Workers',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                hovermode="x unified",
                                height=500
                            )
                            
                            # Display the chart
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add a download button for the comparison data
                            csv_data = chart_data.to_csv(index=False)
                            st.download_button(
                                label="Download Comparison Data",
                                data=csv_data,
                                file_name=f"{dataset_name}_{selected_work_type_for_chart}_comparison.csv",
                                mime="text/csv",
                                key=f"{dataset_name}_{selected_work_type_for_chart}_download"
                            )
                        else:
                            st.info(f"No actual data available for {selected_work_type_for_chart} during the selected period.")
                    else:
                        st.warning("Please select a Punch Code to display the chart.")
                
                except Exception as e:
                    st.error(f"Error during backtesting: {str(e)}")
                    logger.error(f"Error during backtesting: {str(e)}")
                    logger.error(traceback.format_exc())

# Run the main function
if __name__ == "__main__":
    main()