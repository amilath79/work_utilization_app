"""
Predictions page for the Work Utilization Prediction app.
Enhanced models for punch codes 206 & 213 using pipeline architecture.
"""
import os
os.environ["STREAMLIT_SERVER_WATCH_CHANGES"] = "false"
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import traceback
import plotly.graph_objects as go

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import EnhancedFeatureTransformer
from utils.prediction import predict_multiple_days
from utils.data_loader import load_enhanced_models
from utils.sql_data_connector import  save_predictions_to_db
from utils.holiday_utils import is_non_working_day
from config import MODELS_DIR, ENHANCED_WORK_TYPES
from utils.display_utils import get_display_name
# Configure logging
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

# Initialize all session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'ts_data' not in st.session_state:
    st.session_state.ts_data = None
if 'enhanced_df' not in st.session_state:
    st.session_state.enhanced_df = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'enhanced_models' not in st.session_state:
    st.session_state.enhanced_models = None
if 'enhanced_metadata' not in st.session_state:
    st.session_state.enhanced_metadata = None
if 'enhanced_features' not in st.session_state:
    st.session_state.enhanced_features = None
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



# if st.session_state.enhanced_models:
#     for work_type, model in st.session_state.enhanced_models.items():
#         st.write(f"Model {work_type} type: {type(model)}")
#         if hasattr(model, 'steps'):
#             st.write(f"  Pipeline steps : {[step[0] for step in model.steps]}")

# Load enhanced data from saved training data (pickle file)
if st.session_state.enhanced_df is None:
    try:
        training_data_path = os.path.join(MODELS_DIR, 'enhanced_training_data.pkl')
        
        if os.path.exists(training_data_path):
            st.session_state.enhanced_df = pd.read_pickle(training_data_path)
            logger.info(f"✅ Enhanced data loaded from pickle: {len(st.session_state.enhanced_df)} records")
        else:
            logger.warning(f"⚠️ Training data file not found: {training_data_path}")
            
    except Exception as e:
        logger.error(f"❌ Failed to load enhanced data: {str(e)}")
        st.session_state.enhanced_df = None

# Load enhanced models
if st.session_state.enhanced_models is None:
    try:
        models, metadata, features = load_enhanced_models()
        
        if models:
            st.session_state.enhanced_models = models
            st.session_state.enhanced_metadata = metadata
            st.session_state.enhanced_features = features
            logger.info(f"✅ Loaded enhanced models: {list(models.keys())}")
        else:
            logger.warning("⚠️ No enhanced models found")
            
    except Exception as e:
        logger.error(f"❌ Failed to load enhanced models: {str(e)}")
        st.session_state.enhanced_models = {}

# def diagnose_training_data(df):
#     """
#     Diagnose potential issues with training data
#     """
#     print("=== TRAINING DATA DIAGNOSIS ===")
    
#     for work_type in df['WorkType'].unique():
#         wt_data = df[df['WorkType'] == work_type]
        
#         print(f"\nWorkType {work_type}:")
#         print(f"  Records: {len(wt_data)}")
#         print(f"  Hours - Mean: {wt_data['Hours'].mean():.2f}")
#         print(f"  Hours - Median: {wt_data['Hours'].median():.2f}")
#         print(f"  Hours - Max: {wt_data['Hours'].max():.2f}")
#         print(f"  Hours - Min: {wt_data['Hours'].min():.2f}")
        
#         # Check for data quality issues
#         zero_count = (wt_data['Hours'] == 0).sum()
#         low_count = (wt_data['Hours'] < 1).sum()
        
#         print(f"  Zero values: {zero_count} ({zero_count/len(wt_data)*100:.1f}%)")
#         print(f"  Values < 1: {low_count} ({low_count/len(wt_data)*100:.1f}%)")

def ensure_data_and_models():
    """Ensure enhanced data and models are loaded for punch codes 206 & 213"""
    
    # Check enhanced data
    if st.session_state.enhanced_df is None:
        st.error("❌ Enhanced data not loaded. Please run train_models2.py first.")
        return False
    
    # Check enhanced models
    if not st.session_state.enhanced_models:
        st.error("❌ Enhanced models not loaded. Please run train_models2.py first.")
        return False
    
    return True

def get_current_user():
    """Get current user for saving predictions"""
    try:
        import getpass
        return getpass.getuser()
    except:
        return "unknown_user"

# def simple_save_predictions(predictions_dict, hours_dict, username):
#     """
#     Simple function to save predictions to database
#     """
#     try:
#         # Prepare data for saving
#         save_data = []
        
#         for date, work_type_predictions in predictions_dict.items():
#             for work_type, predicted_value in work_type_predictions.items():
#                 hours_value = hours_dict.get(date, {}).get(work_type, predicted_value)
                
#                 save_data.append({
#                     'Date': date,
#                     'WorkType': work_type,
#                     'PredictedHours': hours_value,
#                     'Username': username,
#                     'CreatedAt': datetime.now()
#                 })
        
#         if save_data:
#             save_df = pd.DataFrame(save_data)
            
#             # Save to database using existing function
#             success = save_predictions_to_db(save_df, username)
            
#             if success:
#                 logger.info(f"✅ Saved {len(save_data)} predictions for user {username}")
#                 return True
#             else:
#                 logger.error("❌ Failed to save predictions to database")
#                 return False
#         else:
#             logger.warning("⚠️ No predictions to save")
#             return False
            
#     except Exception as e:
#         logger.error(f"❌ Error saving predictions: {str(e)}")
#         return False

def simple_save_predictions(predictions_dict, hours_dict, username):
    """
    Simple function to save predictions to database
    """
    try:
        # ✅ CORRECT: Pass the dictionaries directly
        success = save_predictions_to_db(predictions_dict, hours_dict, username)
        
        if success:
            logger.info(f"✅ Saved predictions for user {username}")
            return True
        else:
            logger.error("❌ Failed to save predictions to database")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error saving predictions: {str(e)}")
        return False
    
    
def create_resource_plan_table(predictions, hours_predictions, work_types, date_range):
    """
    Create structured data for resource planning pivot table
    FIXED: Properly handles working day logic and tuple returns
    """
    resource_data = []
    
    for date in date_range:
        for work_type in work_types:
            # FIXED: Properly handle working day check
            from utils.holiday_utils import is_working_day_for_punch_code
            is_working_result = is_working_day_for_punch_code(date, work_type)
            
            # Handle both tuple and boolean returns
            if isinstance(is_working_result, tuple):
                is_working, reason = is_working_result
            else:
                is_working = is_working_result
                reason = None
            
            if is_working:
                # Working day - use predicted values
                predicted_hours = predictions.get(date, {}).get(work_type, 0)
                hours_value = hours_predictions.get(date, {}).get(work_type, predicted_hours)
                workers_value = max(1, round(hours_value / 8.0)) if hours_value > 0 else 0
            else:
                # Non-working day - force to 0 (OVERRIDE any predictions)
                hours_value = 0
                workers_value = 0
            
            # Add rows for different metrics
            resource_data.extend([
                {
                    'Date': date,
                    'Day': date.strftime('%A'),
                    'PunchCode': work_type,
                    'Metric': 'Hours',
                    'Value': round(hours_value, 1)
                },
                {
                    'Date': date,
                    'Day': date.strftime('%A'),
                    'PunchCode': work_type,
                    'Metric': 'Workers',
                    'Value': workers_value
                }
            ])
    
    return pd.DataFrame(resource_data)

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
    
    
    # Get available work types from enhanced models
    if st.session_state.enhanced_models:
        available_work_types = st.session_state.enhanced_models 
    else:
        available_work_types = []
        st.warning("⚠️ No models loaded.")
        return
    

    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        # Use enhanced_df for date calculation
        if st.session_state.enhanced_df is not None:
            # Export for debugging (optional - can be removed)
            try:
                st.session_state.enhanced_df.to_excel('enhanced_df.xlsx')
            except Exception as e:
                logger.warning(f"Could not export enhanced_df: {str(e)}")
            
            latest_date = st.session_state.enhanced_df['Date'].max().date()
        else:
            # Fallback if enhanced_df not available
            latest_date = datetime.now().date()
        
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


    # if st.button("Test Pipeline Prediction"):
    #     next_date, preds, hours = predict_next_day(
    #         st.session_state.enhanced_df,
    #         st.session_state.enhanced_models
    #     )
    #     st.write(f"Predictions for {next_date}:")
    #     for wt, hrs in hours.items():
    #         st.write(f"  {wt}: {hrs:.1f} hours")


    # # Debug feature mismatch
    # if st.button("Debug Feature Names"):
    #     # Pick one model to test
    #     work_type = '206'  # Or any work type
    #     pipeline = st.session_state.enhanced_models[work_type]
        
    #     # Check what features the RandomForest expects
    #     lgb_model = pipeline.named_steps['model']
    #     st.write("**Model expects these features:**")
    #     st.write(lgb_model.feature_names_in_[:10]) # First 10 features
        
    #     # Create sample data and check what transformer produces
    #     sample_data = st.session_state.enhanced_df[
    #         st.session_state.enhanced_df['WorkType'] == work_type
    #     ].tail(50)
        
    #     # Transform and check features
    #     transformer = pipeline.named_steps['feature_engineering']
    #     transformed = transformer.transform(sample_data)
    #     st.write("\n**Transformer creates these features:**")
    #     st.write(transformed.columns.tolist()[:10])  # First 10 features

    # Button to trigger prediction
    if st.button("Generate Predictions", type="primary"):
        if not selected_work_types:
            st.warning("Please select at least one work type")
        else:
            # Clear any existing success message when generating new predictions
            st.session_state.save_success_message = None
            
            with st.spinner(f"Generating predictions for {num_days} days..."):
                # Filter enhanced models to selected work types
                filtered_models = {wt: st.session_state.enhanced_models[wt] for wt in selected_work_types if wt in st.session_state.enhanced_models}
                
                if not filtered_models:
                    st.error("No enhanced models available for the selected work types")
                    return
                
                try:
                    # Use enhanced_df for pipeline models
                    prediction_data = st.session_state.enhanced_df
                    
                    # Unpack all three return values from predict_multiple_days
                    predictions, hours_predictions, holiday_info = predict_multiple_days(
                        prediction_data,
                        filtered_models,
                        start_date=pred_start_date,
                        num_days=num_days,
                        use_neural_network=False  # Pipeline models don't need this
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
                # Only set to 0 if it's a non-working day
                display_value = 0 if is_non_working else value
                
                results_records.append({
                    'Date': date,
                    'Work Type': work_type,
                    'Predicted Hours': round(display_value, 1),
                    'Raw Prediction': round(value, 1),
                    'Day of Week': date.strftime('%A'),
                    'Is Non-Working Day': "Yes" if is_non_working else "No",
                    'Reason': reason if is_non_working else ""
                })
        
        results_df = pd.DataFrame(results_records)
        
        # # Display results
        # model_type_text = "Enhanced Pipeline"
        
        # Reconstruct date range for display
        first_date = min(predictions.keys())
        last_date = max(predictions.keys())

        # st.subheader(f"Predictions from {first_date.strftime('%B %d, %Y')} to {last_date.strftime('%B %d, %Y')} using {model_type_text}")
        
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

        # Username input and save button - only show when predictions exist
        st.subheader("Save Predictions")
        username = get_current_user()

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