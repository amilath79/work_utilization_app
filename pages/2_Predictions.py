"""
Predictions page for the Work Utilization Prediction app.
"""
import os
os.environ["STREAMLIT_SERVER_WATCH_CHANGES"] = "false"
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import io
import sys
import traceback
import plotly.graph_objects as go

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import engineer_features, create_lag_features
from utils.prediction import predict_next_day, predict_multiple_days, evaluate_predictions
from utils.data_loader import export_predictions, load_models, load_data
from utils.sql_data_connector import extract_sql_data, save_predictions_to_db
from utils.holiday_utils import is_non_working_day
from utils.sql_data_connector import extract_sql_data
from config import MODELS_DIR, DATA_DIR, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION

# Configure logging to display debug information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Workforce Predictions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
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
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None
if 'current_hours_predictions' not in st.session_state:
    st.session_state.current_hours_predictions = None
if 'save_button_clicked' not in st.session_state:
    st.session_state.save_button_clicked = False
if 'save_success_message' not in st.session_state:
    st.session_state.save_success_message = None

def set_save_clicked():
    st.session_state.save_button_clicked = True

def load_workutilizationdata():
    """Load data from the WorkUtilizationData table"""
    try:
        sql_query = """
        SELECT Date, PunchCode as WorkType, Hours, NoOfMan, SystemHours, Quantity, ResourceKPI, SystemKPI 
        FROM WorkUtilizationData WHERE PunchCode IN (215, 209, 213, 211, 214, 202, 203, 206, 208, 210, 217) AND Hours <> 0
        ORDER BY Date
        """
        
        with st.spinner("Loading work utilization data..."):
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
                
                logger.info(f"Loaded {len(df)} records from WorkUtilizationData")
                return df
            else:
                logger.warning("No data returned from WorkUtilizationData")
                return None
    except Exception as e:
        logger.error(f"Error loading data from WorkUtilizationData: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def simple_save_predictions(predictions_dict, hours_dict, username, server=None, database=None):
    """
    Simple function to save predictions to database with minimal complexity
    """
    import pyodbc
    from config import SQL_SERVER, SQL_DATABASE
    
    # Use provided or default values
    server = server or SQL_SERVER
    database = database or SQL_DATABASE
    
    try:
        # Create connection string
        conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
        
        # Connect to database
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Track success
        saved_count = 0
        
        # Process each prediction one by one
        for date, work_types in predictions_dict.items():
            for work_type, man_value in work_types.items():
                try:
                    # Convert work_type to integer
                    punch_code = int(work_type)
                    
                    # Format date as string
                    date_str = date.strftime('%Y-%m-%d')
                    
                    # Get hours value
                    hours = 0.0
                    if date in hours_dict and work_type in hours_dict[date]:
                        hours = float(hours_dict[date][work_type])
                    
                    # Execute stored procedure
                    cursor.execute(
                        "EXEC usp_UpsertPrediction @Date=?, @PunchCode=?, @NoOfMan=?, @Hours=?, @Username=?",
                        date_str, 
                        punch_code, 
                        float(man_value), 
                        hours, 
                        username
                    )
                    
                    # Commit immediately
                    conn.commit()
                    saved_count += 1
                    
                except Exception as e:
                    print(f"Error saving prediction for {date_str}, {work_type}: {str(e)}")
                    # Continue with next prediction
        
        # Close resources
        cursor.close()
        conn.close()
        
        return saved_count > 0
    
    except Exception as e:
        print(f"Database error: {str(e)}")
        return False

def ensure_data_and_models():
    """Ensure data and models are loaded"""
    # Try to load data if not already loaded
    if st.session_state.df is None:
        st.session_state.df = load_workutilizationdata()
    
    # If database load failed, offer Excel options
    if st.session_state.df is None:
        st.error("Could not load data from database. Please upload Excel file instead.")
        
        uploaded_file = st.file_uploader(
            "Upload Work Utilization Excel File", 
            type=["xlsx", "xls"],
            help="Upload Excel file with work utilization data"
        )
        
        if uploaded_file is not None:
            st.session_state.df = load_data(uploaded_file)
        
        use_sample_data = st.checkbox("Use Sample Data", value=False)
        
        if use_sample_data:
            sample_path = os.path.join(DATA_DIR, "sample_work_utilization.xlsx")
            
            if os.path.exists(sample_path):
                st.session_state.df = load_data(sample_path)
        
        # Check if we have data after trying upload options
        if st.session_state.df is None:
            st.warning("No data available. Please upload a file or connect to the database.")
            return False
    
    # Process the dataset
    if st.session_state.df is not None and st.session_state.processed_df is None:
        with st.spinner("Processing data..."):
            st.session_state.processed_df = engineer_features(st.session_state.df)
            st.session_state.ts_data = create_lag_features(st.session_state.processed_df)
    
    # Load models
    if st.session_state.models is None:
        with st.spinner("Loading models..."):
            models, feature_importances, metrics = load_models()
            
            if models:
                st.session_state.models = models
                st.session_state.feature_importances = feature_importances
                st.session_state.metrics = metrics
                logger.info(f"Successfully loaded {len(models)} models: {list(models.keys())}")
            else:
                st.error("No trained models available. Please train models first.")
                return False
    
    return True


def create_resource_plan_table(predictions_dict, hours_dict, selected_work_types, dates):
    """Create a structured resource plan table with all selected work types"""
    # Prepare data in the format needed for the pivot table
    data = []
    
    for date in dates:
        # Check if day is non-working
        is_non_working, reason = is_non_working_day(date)
        
        # Add debug logging
        logger.info(f"Processing date {date}: Is non-working: {is_non_working}, Reason: {reason}")
        logger.info(f"Available predictions for this date: {date in predictions_dict}")
        
        for wt in selected_work_types:
            # Default values
            man_value = 0
            hours_value = 0
            
            # Only use zero for non-working days
            if not is_non_working:
                # For working days, use the predicted values from the dictionaries
                if date in predictions_dict and wt in predictions_dict[date]:
                    man_value = predictions_dict[date][wt]
                    logger.info(f"Found prediction for {date}, {wt}: {man_value}")
                    
                    # Also get hours value
                    if date in hours_dict and wt in hours_dict[date]:
                        hours_value = hours_dict[date][wt]
                    else:
                        # Default to 8 hours per person if not in hours_dict
                        hours_value = man_value * 8.0
                else:
                    logger.warning(f"No prediction found for {date}, {wt}")
            
            # Add entry for NoOfMan
            data.append({
                'Date': date,
                'PunchCode': wt,
                'Metric': 'NoOfMan',
                'Value': round(man_value) if pd.notnull(man_value) else 0,
                'Day': date.strftime('%a')
            })
            
            # Add entry for Hours
            data.append({
                'Date': date,
                'PunchCode': wt,
                'Metric': 'Hours',
                'Value': round(hours_value, 1) if pd.notnull(hours_value) else 0,
                'Day': date.strftime('%a')
            })
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    return df


def get_current_user():
    """Get the current user from the system"""
    try:
        import os
        import getpass
        username = os.environ.get('USERNAME', getpass.getuser())
        return username
    except:
        return "unknown"


def main():
    st.header("Workforce Predictions")
    
    # Display save success message if it exists
    if st.session_state.save_success_message:
        st.success(st.session_state.save_success_message)
        # Clear the message after displaying it
        st.session_state.save_success_message = None
    
    # Check data and models
    if not ensure_data_and_models():
        return
    
    # Get available work types from models
    available_work_types = list(st.session_state.models.keys())
    logger.info(f"Available models: {available_work_types}")
    
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
        st.warning("Neural network models are not available. Using Random Forest models instead.")
        use_neural = False
        model_type = "Random Forest"
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        # Find the latest date from the dataset
        latest_date = st.session_state.ts_data['Date'].max().date()
        next_date = latest_date + timedelta(days=1)
        
        pred_start_date = st.date_input(
            "Start Date",
            value=next_date,
            min_value=latest_date,
            disabled=True,
            help="Select the start date for the prediction period"
        )
    
    with col2:
        # Number of days to predict
        num_days = st.slider(
            "Number of Days",
            min_value=1,
            max_value=365,
            value=7,
            help="Select the number of days to predict"
        )
        
        pred_end_date = pred_start_date + timedelta(days=num_days-1)
        st.write(f"End Date: {pred_end_date.strftime('%Y-%m-%d')}")
    
    # Work type selector
    selected_work_types = st.multiselect(
        "Select Punch Codes",
        options=available_work_types,
        default=available_work_types,
        help="Select the work types for which you want to make predictions"
    )
    
    # Button to trigger prediction
    if st.button("Generate Predictions", type="primary"):
        if not selected_work_types:
            st.warning("Please select at least one work type")
        else:
            # Clear any existing success message when generating new predictions
            st.session_state.save_success_message = None
            
            with st.spinner(f"Generating predictions for {num_days} days..."):
                # Filter models to selected work types
                filtered_models = {wt: st.session_state.models[wt] for wt in selected_work_types if wt in st.session_state.models}
                
                if not filtered_models:
                    st.error("No models available for the selected work types")
                    return
                
                try:
                    # Unpack all three return values from predict_multiple_days
                    predictions, hours_predictions, holiday_info = predict_multiple_days(
                        st.session_state.ts_data,
                        filtered_models,
                        num_days=num_days,
                        use_neural_network=use_neural
                    )
                    
                    # Store predictions in session state for later use
                    st.session_state.current_predictions = predictions
                    st.session_state.current_hours_predictions = hours_predictions
                    
                    if not predictions:
                        st.error("Failed to generate predictions")
                        return
                    
                    st.success("✅ Predictions generated successfully!")
                        
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    logger.error(f"Error generating predictions: {str(e)}")
                    logger.error(traceback.format_exc())

    # Only show predictions and save options if predictions exist
    if st.session_state.current_predictions:
        predictions = st.session_state.current_predictions
        hours_predictions = st.session_state.current_hours_predictions
        
        # Create a dataframe for display
        results_records = []
        
        for date, pred_dict in predictions.items():
            # Check if day is non-working
            is_non_working, reason = is_non_working_day(date)
            
            for work_type, value in pred_dict.items():
                # IMPORTANT: Only set to 0 if it's a non-working day
                # Otherwise use the actual predicted value
                display_value = 0 if is_non_working else value
                
                results_records.append({
                    'Date': date,
                    'Work Type': work_type,
                    'Predicted Workers': round(display_value),
                    'Raw Prediction': round(value, 1),  # Show original prediction for debugging
                    'Day of Week': date.strftime('%A'),
                    'Is Non-Working Day': "Yes" if is_non_working else "No",
                    'Reason': reason if is_non_working else ""
                })
        
        results_df = pd.DataFrame(results_records)
        
        # Display results
        model_type_text = "Neural Network" if use_neural else "Random Forest"
        
        # Reconstruct date range for display
        first_date = min(predictions.keys())
        last_date = max(predictions.keys())

        st.subheader(f"Predictions from {first_date.strftime('%B %d, %Y')} to {last_date.strftime('%B %d, %Y')} using {model_type_text}")
        
        # Holiday information
        with st.expander("📊 View Holiday Information", expanded=False):
            # If there are non-working days in the prediction period, show a warning
            non_working_dates = results_df[results_df['Is Non-Working Day'] == "Yes"]['Date'].unique()
            if len(non_working_dates) > 0:
                st.warning("⚠️ Non-working days detected during the prediction period:")
                for non_working_date in non_working_dates:
                    reason = results_df[results_df['Date'] == non_working_date]['Reason'].iloc[0]
                    st.info(f"• {non_working_date.strftime('%A, %B %d, %Y')}: {reason} (No work carried out)")
        
        # Add pivot table for resource planning
        st.subheader("Resource Planning View")

        # Generate the date range
        date_range = list(predictions.keys())
        selected_work_types_from_predictions = list(set(wt for pred_dict in predictions.values() for wt in pred_dict.keys()))
        
        # Create structured data with all selected work types
        resource_data = create_resource_plan_table(
            predictions, 
            hours_predictions, 
            selected_work_types_from_predictions,
            date_range
        )
        
        # Create daily pivot table with punch codes as columns and metrics as sub-columns
        daily_pivot = pd.pivot_table(
            resource_data,
            values='Value',
            index=['Date', 'Day'],
            columns=['PunchCode', 'Metric'],
            fill_value=0
        )
        
        # Create monthly pivot table
        resource_data['Month'] = resource_data['Date'].dt.strftime('%Y-%m')
        monthly_pivot = pd.pivot_table(
            resource_data,
            values='Value',
            index='Month',
            columns=['PunchCode', 'Metric'],
            fill_value=0,
            aggfunc='sum'
        )
        
        # Display pivot tables
        st.write("### Daily Resource Plan")
        st.dataframe(daily_pivot, use_container_width=True)
        
        st.write("### Monthly Resource Plan")
        st.dataframe(monthly_pivot, use_container_width=True)

        # # Download options
        # st.subheader("Download Options")
        # col1, col2, col3 = st.columns(3)
        
        # with col1:
        #     # Download raw predictions
        #     csv_data = results_df.to_csv(index=False)
        #     st.download_button(
        #         label="Download Predictions (CSV)",
        #         data=csv_data,
        #         file_name=f"workforce_predictions_{model_type_text.replace(' ', '_')}_{first_date.strftime('%Y%m%d')}_to_{last_date.strftime('%Y%m%d')}.csv",
        #         mime="text/csv"
        #     )
        
        # with col2:
        #     # Download resource plan
        #     pivot_buffer = io.BytesIO()
        #     daily_pivot.to_excel(pivot_buffer)
        #     pivot_buffer.seek(0)

        #     st.download_button(
        #         label="Download Daily Resource Plan (Excel)",
        #         data=pivot_buffer,
        #         file_name=f"daily_resource_plan_{first_date.strftime('%Y%m%d')}.xlsx",
        #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        #     )

        # with col3:
        #     # Download monthly plan
        #     pivot_buffer_month = io.BytesIO()
        #     monthly_pivot.to_excel(pivot_buffer_month)
        #     pivot_buffer_month.seek(0)

        #     st.download_button(
        #         label="Download Monthly Resource Plan (Excel)",
        #         data=pivot_buffer_month,
        #         file_name=f"monthly_resource_plan_{first_date.strftime('%Y%m%d')}.xlsx",
        #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        #     )

        # Username input and save button - only show when predictions exist
        st.subheader("Save Predictions")
        username = value=get_current_user()

        # Save button
        if st.button("Save Predictions to Database", type="primary"):
            if not username:
                st.error("Please enter your username")
            else:
                with st.spinner("Saving predictions..."):
                    try:
                        # Use the simple save function
                        success = simple_save_predictions(
                            predictions_dict=st.session_state.current_predictions,
                            hours_dict=st.session_state.current_hours_predictions,
                            username=username
                        )
                        
                        if success:
                            # Store success message in session state
                            st.session_state.save_success_message = "✅ Predictions saved successfully!"
                            # Clear predictions from session state to hide the dataframe
                            st.session_state.current_predictions = None
                            st.session_state.current_hours_predictions = None
                            # Force a rerun to update the display
                            st.rerun()
                        else:
                            st.error("Failed to save predictions")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()