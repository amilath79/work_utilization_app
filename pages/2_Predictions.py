"""
Predictions page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import io
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import engineer_features, create_lag_features
from utils.prediction import predict_next_day, predict_multiple_days, evaluate_predictions
from utils.data_loader import export_predictions, load_models, load_data
from utils.holiday_utils import is_non_working_day
from config import MODELS_DIR, DATA_DIR

# Configure page
st.set_page_config(
    page_title="Workforce Predictions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

# Create session state for data persistence if not present
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
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'prediction_date' not in st.session_state:
    st.session_state.prediction_date = None

def load_work_types():
    try:
        # Path to the Excel file
        excel_path = os.path.join(DATA_DIR, 'WorkTypes.xlsx')
        
        if os.path.exists(excel_path):
            # Load work types from Excel file
            work_types_df = pd.read_excel(excel_path)
            
            # Convert to a format usable by the multiselect
            available_work_types = [f"{row['Name']}" for _, row in work_types_df.iterrows()]
            available_work_types = sorted(available_work_types)
        else:
            st.error(f"Work types file not found at: {excel_path}")
            available_work_types = []
    except Exception as e:
        st.error(f"Error loading work types: {e}")
        available_work_types = []
    return available_work_types

# Check if we have data and models
def ensure_data_and_models():
    # Check if we have data
    if st.session_state.df is None:
        # Display data loading options
        st.error("No data loaded. Please load data first.")
        
        uploaded_file = st.file_uploader(
            "Upload Excel File", 
            type=["xlsx", "xls"],
            help="Upload your Work Utilization Excel file"
        )
        
        use_sample_data = st.checkbox(
            "Use Sample Data", 
            value=False,
            help="Use sample data if you don't have your own file"
        )
        
        if uploaded_file is not None:
            # Use uploaded file
            st.session_state.df = load_data(uploaded_file)
            st.rerun()
        elif use_sample_data:
            # Use sample data
            sample_path = os.path.join(DATA_DIR, "sample_work_utilization.xlsx")
            
            if os.path.exists(sample_path):
                st.session_state.df = load_data(sample_path)
                st.rerun()
            else:
                st.warning("Sample data file not found. Please upload your own data.")
                return False
        return False
    
    # Process data if not already done
    if 'processed_df' not in st.session_state or st.session_state.processed_df is None:
        with st.spinner("Processing data for prediction..."):
            st.session_state.processed_df = engineer_features(st.session_state.df)
    
    if 'ts_data' not in st.session_state or st.session_state.ts_data is None:
        with st.spinner("Creating time series features..."):
            st.session_state.ts_data = create_lag_features(st.session_state.processed_df)
    
    # Check if we have models
    if 'models' not in st.session_state or st.session_state.models is None:
        with st.spinner("Loading models..."):
            models, feature_importances, metrics = load_models()
            
            if models:
                st.session_state.models = models
                st.session_state.feature_importances = feature_importances
                st.session_state.metrics = metrics
            else:
                st.error("No trained models available. Please train models first.")
                return False
    
    return True

def main():
    st.header("Workforce Predictions")
    
    # Check data and models
    if not ensure_data_and_models():
        return
    
    # Get available work types
    available_work_types = load_work_types()
    
    # Prediction options
    st.subheader("Prediction Options")
    
    # Add model selection option
    model_type = st.radio(
        "Select Model Type",
        ["Random Forest", "Neural Network", "Ensemble (Both)"],
        horizontal=True,
        help="Choose which model to use for predictions"
    )
    
    # Set use_neural parameter based on selection
    use_neural = model_type in ["Neural Network", "Ensemble (Both)"]
    
    # Check if neural network models are available
    nn_available = False
    try:
        nn_path = os.path.join(MODELS_DIR, "work_utilization_nn_models.pkl")
        nn_available = os.path.exists(nn_path)
    except:
        pass
    
    if use_neural and not nn_available:
        st.warning("Neural network models are not available. Please run training for neural network models first. Using Random Forest models instead.")
        use_neural = False
        model_type = "Random Forest"
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        # Date range selector
        latest_date = st.session_state.df['Date'].max().date()
        next_date = latest_date + timedelta(days=1)
        
        pred_start_date = st.date_input(
            "Start Date",
            value=next_date,
            min_value=latest_date,
            help="Select the start date for the prediction period"
        )
    
    with col2:
        # Number of days to predict
        num_days = st.slider(
            "Number of Days",
            min_value=1,
            max_value=30,
            value=7,
            help="Select the number of days to predict"
        )
        
        pred_end_date = pred_start_date + timedelta(days=num_days-1)
        st.write(f"End Date: {pred_end_date.strftime('%Y-%m-%d')}")
    
    # Work type selector
    selected_work_types = st.multiselect(
        "Select Work Types",
        options=available_work_types,
        default=available_work_types[:10] if len(available_work_types) > 10 else available_work_types,
        help="Select the work types for which you want to make predictions"
    )
    
    # Button to trigger prediction
    if st.button("Generate Predictions", type="primary"):
        if not selected_work_types:
            st.warning("Please select at least one work type")
        else:
            with st.spinner(f"Generating predictions for {num_days} days..."):
                # Filter models to selected work types
                filtered_models = {wt: st.session_state.models[wt] for wt in selected_work_types if wt in st.session_state.models}
                
                # Predict multiple days
                multi_day_predictions, holiday_info = predict_multiple_days(
                    st.session_state.ts_data,
                    filtered_models,
                    num_days=num_days,
                    use_neural_network=use_neural
                )
                
                # Store in session state
                st.session_state.last_prediction = multi_day_predictions
                
                # Display results
                model_type_text = "Neural Network" if use_neural else "Random Forest"
                st.subheader(f"Predictions from {pred_start_date.strftime('%B %d, %Y')} to {pred_end_date.strftime('%B %d, %Y')} using {model_type_text}")
                
                # Create a dataframe for display
                results_records = []
                for date, predictions in multi_day_predictions.items():
                    # Explicitly check if day is non-working using utility function
                    is_non_working, reason = is_non_working_day(date)
                    
                    for work_type, value in predictions.items():
                        if work_type in selected_work_types:
                            # Set value to 0 only if it's a non-working day
                            # This preserves predicted values for Sundays
                            display_value = 0 if is_non_working else value
                            
                            results_records.append({
                                'Date': date,
                                'Work Type': work_type,
                                'Predicted Workers': round(display_value, 0),
                                'Day of Week': date.strftime('%A'),
                                'Is Non-Working Day': "Yes" if is_non_working else "No",
                                'Reason': reason if is_non_working else ""
                            })
                
                results_df = pd.DataFrame(results_records)
                
                # If there are non-working days in the prediction period, show a warning
                non_working_dates = results_df[results_df['Is Non-Working Day'] == "Yes"]['Date'].unique()
                if len(non_working_dates) > 0:
                    st.warning("⚠️ Non-working days detected during the prediction period:")
                    for non_working_date in non_working_dates:
                        reason = results_df[results_df['Date'] == non_working_date]['Reason'].iloc[0]
                        st.info(f"• {non_working_date.strftime('%A, %B %d, %Y')}: {reason} (No work carried out)")
                
                # Display table
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    column_config={
                        'Date': st.column_config.DateColumn('Date'),
                        'Predicted Workers': st.column_config.NumberColumn(
                            'Predicted Workers',
                            format="%.2f",
                            help="Predicted number of workers required"
                        ),
                        'Is Non-Working Day': st.column_config.TextColumn(
                            'Is Non-Working Day',
                            help="Whether this date is a non-working day"
                        ),
                        'Reason': st.column_config.TextColumn(
                            'Reason',
                            help="Reason for non-working day"
                        )
                    }
                )

                # Add pivot table for resource planning
                st.subheader("Resource Planning View")

                # Create a simpler pivot table
                pivot = pd.pivot_table(
                    results_df,
                    values='Predicted Workers',
                    index='Date',
                    columns='Work Type',
                    fill_value=0
                )

                # Add day of week for better context
                pivot['Day'] = [d.strftime('%a') for d in pivot.index]  # %a gives abbreviated day name

                # Move Day column to first position
                cols = pivot.columns.tolist()
                cols.insert(0, cols.pop(cols.index('Day')))
                pivot = pivot[cols]

                # Display the pivot table
                st.dataframe(pivot, use_container_width=True)

                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download raw predictions
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions (CSV)",
                        data=csv_data,
                        file_name=f"workforce_predictions_{model_type}_{pred_start_date.strftime('%Y%m%d')}_to_{pred_end_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download resource plan
                    pivot_buffer = io.BytesIO()
                    pivot.to_excel(pivot_buffer)
                    pivot_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Resource Plan (Excel)",
                        data=pivot_buffer,
                        file_name=f"resource_plan_{pred_start_date.strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    # Add backtesting section
    st.markdown("---")
    st.subheader("Model Validation & Backtesting")
    
    # Explanation
    st.write("""
    Backtesting allows you to evaluate how well the model would have performed in the past by comparing
    predictions against actual historical data.
    """)
    
    # Date range for backtesting
    col1, col2 = st.columns(2)
    
    with col1:
        backtest_days = st.slider(
            "Number of Days for Backtesting",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="Select number of days from the end of your dataset to use for validation"
        )
    
    with col2:
        backtest_work_types = st.multiselect(
            "Select Work Types for Backtesting",
            options=available_work_types,
            default=[available_work_types[0]] if available_work_types else [],
            help="Select work types to include in backtesting",
            key="backtest_work_types"
        )
    
    # Add neural network option for backtesting
    backtest_model_type = st.radio(
        "Select Model Type for Backtesting",
        ["Random Forest", "Neural Network"],
        horizontal=True,
        help="Choose which model to use for backtesting",
        key="backtest_model"
    )
    
    use_neural_backtest = backtest_model_type == "Neural Network"
    
    if st.button("Run Backtesting", type="primary"):
        if not backtest_work_types:
            st.warning("Please select at least one work type for backtesting")
        else:
            if use_neural_backtest and not nn_available:
                st.warning("Neural network models are not available. Using Random Forest models for backtesting.")
                use_neural_backtest = False
            
            with st.spinner(f"Running backtesting for the last {backtest_days} days..."):
                # Filter models to selected work types
                filtered_models = {wt: st.session_state.models[wt] for wt in backtest_work_types if wt in st.session_state.models}
                
                # Get the data for backtesting
                max_date = st.session_state.df['Date'].max()
                backtest_start = max_date - timedelta(days=backtest_days)
                
                backtest_data = st.session_state.ts_data[st.session_state.ts_data['Date'] <= backtest_start]
                actual_data = st.session_state.ts_data[
                    (st.session_state.ts_data['Date'] > backtest_start) & 
                    (st.session_state.ts_data['Date'] <= max_date)
                ]
                
                # Generate predictions for the backtest period
                backtest_predictions, holiday_info = predict_multiple_days(
                    backtest_data,
                    filtered_models,
                    num_days=backtest_days,
                    use_neural_network=use_neural_backtest
                )
                
                # Create a dataframe with predictions and actuals
                results_records = []
                metrics_by_worktype = {}
                
                for date, predictions in backtest_predictions.items():
                    # Explicitly check if day is non-working using utility function
                    is_non_working, reason = is_non_working_day(date)
                    
                    for work_type, pred_value in predictions.items():
                        if work_type in backtest_work_types:
                            # Use actual predicted value for Sundays
                            # Only set to 0 for non-working days (Saturdays and holidays)
                            display_pred = 0 if is_non_working else pred_value
                            
                            # Get actual value
                            actual_records = actual_data[
                                (actual_data['Date'] == date) & 
                                (actual_data['WorkType'] == work_type)
                            ]
                            
                            actual_value = actual_records['NoOfMan'].sum() if not actual_records.empty else None
                            
                            results_records.append({
                                'Date': date,
                                'Work Type': work_type,
                                'Predicted': display_pred,
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
                
                # If there are non-working days in the backtesting period, show info
                non_working_dates = results_df[results_df['Is Non-Working Day'] == "Yes"]['Date'].unique()
                if len(non_working_dates) > 0:
                    st.info("📅 The backtesting period includes non-working days (Saturdays and Swedish holidays) where no work is carried out. Predictions for these days are set to 0.")
                
                # Display metrics
                st.write("Model Performance Metrics:")
                
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
                
                # Display results table
                st.subheader("Prediction Results vs Actual Values")
                
                # IMPORTANT: Check if there are Sunday results to validate our fix
                sunday_data = results_df[results_df['Day of Week'] == 'Sunday']
                if not sunday_data.empty:
                    st.write("### Sunday Results Check")
                    
                    # Filter to show only Sundays to verify resource allocation
                    sunday_results = sunday_data.sort_values('Date')
                    st.dataframe(
                        sunday_results,
                        use_container_width=True,
                        column_config={
                            'Date': st.column_config.DateColumn('Date'),
                            'Day of Week': st.column_config.TextColumn('Day of Week'),
                            'Predicted': st.column_config.NumberColumn('Predicted', format="%.2f"),
                            'Actual': st.column_config.NumberColumn('Actual', format="%.2f")
                        }
                    )
                
                # Display all results
                st.write("### All Prediction Results")
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    column_config={
                        'Date': st.column_config.DateColumn('Date'),
                        'Day of Week': st.column_config.TextColumn('Day of Week'),
                        'Predicted': st.column_config.NumberColumn('Predicted', format="%.2f"),
                        'Actual': st.column_config.NumberColumn('Actual', format="%.2f"),
                        'Difference': st.column_config.NumberColumn('Difference', format="%.2f")
                    }
                )

# Run the main function
if __name__ == "__main__":
    main()