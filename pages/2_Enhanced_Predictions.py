"""
Enhanced Predictions page for the Work Utilization Prediction app.
Uses enhanced models from train_models2.py for punch codes 206 & 213.
Follows enterprise coding patterns with clear, maintainable structure.
"""
import os
os.environ["STREAMLIT_SERVER_WATCH_CHANGES"] = "false"
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import io
import sys
import traceback
import plotly.graph_objects as go
import pickle
import json
import glob

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import add_rolling_features_by_group, add_lag_features_by_group, add_cyclical_features, add_trend_features, add_pattern_features
from utils.sql_data_connector import extract_sql_data, save_predictions_to_db
from utils.holiday_utils import is_non_working_day, is_working_day_for_punch_code
from config import MODELS_DIR, DATA_DIR, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, FEATURE_GROUPS, DEFAULT_MODEL_PARAMS

# Configure logging (follow enterprise patterns)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("enhanced_predictions")

# Configure page
st.set_page_config(
    page_title="Enhanced Predictions (206 & 213)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'enhanced_df' not in st.session_state:
    st.session_state.enhanced_df = None
if 'enhanced_models' not in st.session_state:
    st.session_state.enhanced_models = None
if 'enhanced_features' not in st.session_state:
    st.session_state.enhanced_features = None
if 'enhanced_metadata' not in st.session_state:
    st.session_state.enhanced_metadata = None
if 'enhanced_predictions' not in st.session_state:
    st.session_state.enhanced_predictions = None

def load_enhanced_training_data():
    """
    Load training data for enhanced models (punch codes 206 & 213)
    Uses same data loading pattern as train_models2.py
    """
    try:
        logger.info("🔗 Loading training data for enhanced prediction")
        
        query = """
        SELECT Date, PunchCode as WorkType, Hours, NoOfMan, SystemHours, NoRows as Quantity, SystemKPI 
        FROM WorkUtilizationData 
        WHERE PunchCode IN (206, 213) 
        AND Hours > 0 
        AND NoOfMan > 0 
        AND SystemHours > 0 
        AND NoRows > 0
        AND Date < '2025-05-01'
        ORDER BY Date
        """
        
        df = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if df is None or df.empty:
            logger.error("❌ No data returned from database")
            return None
            
        # Data preprocessing (follow existing patterns)
        df['Date'] = pd.to_datetime(df['Date'])
        df['WorkType'] = df['WorkType'].astype(str)
        
        # Handle decimals appropriately
        df['NoOfMan'] = df['NoOfMan'].round(0).astype(int)
        df['SystemHours'] = df['SystemHours'].round(1)
        df['SystemKPI'] = df['SystemKPI'].round(2)
        df['Hours'] = df['Hours'].round(1)
        df['Quantity'] = df['Quantity'].round(0).astype(int)
        
        logger.info(f"✅ Loaded {len(df)} records for prediction")
        logger.info(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"📊 Punch codes: {df['WorkType'].unique()}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading training data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def add_productivity_features(df):
    """
    Add productivity-related features following train_models2.py patterns
    Clean, efficient implementation
    """
    try:
        df = df.copy()
        
        # Core productivity metrics
        df['HoursPerMan'] = df['Hours'] / np.maximum(df['NoOfMan'], 1)
        df['SystemHoursPerMan'] = df['SystemHours'] / np.maximum(df['NoOfMan'], 1)
        df['QuantityPerMan'] = df['Quantity'] / np.maximum(df['NoOfMan'], 1)
        df['QuantityPerHour'] = df['Quantity'] / np.maximum(df['Hours'], 1)
        
        # Efficiency ratios
        df['ActualVsSystemHours'] = df['Hours'] / np.maximum(df['SystemHours'], 1)
        df['EfficiencyRatio'] = df['SystemKPI'] / np.maximum(df['HoursPerMan'], 1)
        
        logger.info("✅ Added productivity features")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error adding productivity features: {str(e)}")
        return df

def create_enhanced_features(df):
    """
    Create enhanced features following train_models2.py patterns
    Maintains code clarity and enterprise structure
    """
    try:
        logger.info("🔧 Starting enhanced feature creation")
        df_enhanced = df.copy()

        # Log enabled feature groups
        enabled = [k for k, v in FEATURE_GROUPS.items() if v]
        logger.info(f"📊 Active Feature Groups: {enabled}")

        # ───── Temporal Features ─────
        if FEATURE_GROUPS.get('DATE_FEATURES'):
            df_enhanced['DayOfWeek'] = df_enhanced['Date'].dt.dayofweek
            df_enhanced['Month'] = df_enhanced['Date'].dt.month
            df_enhanced['WeekNo'] = df_enhanced['Date'].dt.isocalendar().week
            df_enhanced['Year'] = df_enhanced['Date'].dt.year
            df_enhanced['Quarter'] = df_enhanced['Date'].dt.quarter

        # ───── Schedule-based Binary Features ─────
        df_enhanced['ScheduleType'] = np.where(df_enhanced['WorkType'] == '206', '6DAY', '5DAY')
        df_enhanced['CanWorkSunday'] = np.where(df_enhanced['WorkType'] == '206', 1, 0)
        df_enhanced['IsSunday'] = (df_enhanced['DayOfWeek'] == 6).astype(int)
        df_enhanced['IsWeekend'] = (df_enhanced['DayOfWeek'] >= 5).astype(int)
        df_enhanced['IsMonday'] = (df_enhanced['DayOfWeek'] == 0).astype(int)
        df_enhanced['IsFriday'] = (df_enhanced['DayOfWeek'] == 4).astype(int)

        # ───── Productivity Features ─────
        if FEATURE_GROUPS.get('PRODUCTIVITY_FEATURES'):
            df_enhanced = add_productivity_features(df_enhanced)

        # ───── Apply Lag/Rolling/Trend/Pattern per WorkType ─────
        for work_type in ['206', '213']:
            work_data = df_enhanced[df_enhanced['WorkType'] == work_type].copy()

            if len(work_data) > 30:
                if FEATURE_GROUPS.get('LAG_FEATURES'):
                    work_data = add_lag_features_by_group(work_data)

                if FEATURE_GROUPS.get('ROLLING_FEATURES'):
                    work_data = add_rolling_features_by_group(work_data)

                if FEATURE_GROUPS.get('TREND_FEATURES'):
                    work_data = add_trend_features(work_data)

                if FEATURE_GROUPS.get('PATTERN_FEATURES'):
                    work_data = add_pattern_features(work_data)

                # ───── Cyclical Features ─────
                if FEATURE_GROUPS.get('CYCLICAL_FEATURES'):
                    work_data = add_cyclical_features(work_data)

                # Update enhanced dataframe
                df_enhanced.loc[df_enhanced['WorkType'] == work_type] = work_data

        logger.info(f"✅ Enhanced feature creation complete - {len(df_enhanced.columns)} total features")
        return df_enhanced

    except Exception as e:
        logger.error(f"❌ Error creating enhanced features: {str(e)}")
        logger.error(traceback.format_exc())
        return df

def load_enhanced_models():
    """
    Load enhanced models, metadata, and features from train_models2.py
    Follows enterprise patterns for model loading
    """
    try:
        logger.info("📂 Loading enhanced models from train_models2.py")
        
        # Find the latest enhanced model files
        model_files = glob.glob(os.path.join(MODELS_DIR, "enhanced_model_*.pkl"))
        metadata_files = glob.glob(os.path.join(MODELS_DIR, "enhanced_models_metadata_*.json"))
        feature_files = glob.glob(os.path.join(MODELS_DIR, "enhanced_features_*.json"))
        
        if not model_files:
            logger.error("❌ No enhanced model files found")
            return None, None, None
        
        # Get the latest files (by timestamp)
        latest_model_file = max(model_files, key=os.path.getctime)
        latest_metadata_file = max(metadata_files, key=os.path.getctime)
        latest_feature_file = max(feature_files, key=os.path.getctime)
        
        # Load models
        with open(latest_model_file, 'rb') as f:
            loaded_models = pickle.load(f)

        # Check if it's a dictionary or single model
        if isinstance(loaded_models, dict):
            models = loaded_models
            logger.info(f"✅ Loaded models dictionary: {list(models.keys())}")
        else:
            # Single Pipeline model - determine work type from filename
            if '206' in latest_model_file:
                models = {'206': loaded_models}
            elif '213' in latest_model_file:
                models = {'213': loaded_models}
            else:
                models = {'206': loaded_models}  # Default assumption
            logger.info(f"✅ Loaded single model: {type(loaded_models).__name__}")
            
            # Load metadata
            with open(latest_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load features
            with open(latest_feature_file, 'r') as f:
                features = json.load(f)
            
            logger.info(f"✅ Loaded enhanced models: {list(models.keys())}")
            logger.info(f"📁 Model file: {os.path.basename(latest_model_file)}")
            
            return models, metadata, features
            
    except Exception as e:
        logger.error(f"❌ Error loading enhanced models: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def create_prediction_features(df, work_type, prediction_date):
    """
    Create features for prediction following enhanced patterns
    Clean implementation without unnecessary complexity
    """
    try:
        # Get work type data and sort by date
        work_data = df[df['WorkType'] == work_type].copy().sort_values('Date')
        
        if len(work_data) < 30:
            logger.warning(f"Insufficient data for {work_type}: {len(work_data)} records")
            return None
        
        # Get the latest record as base
        latest_record = work_data.iloc[-1:].copy()
        
        # Update date and temporal features
        latest_record['Date'] = prediction_date
        latest_record['DayOfWeek'] = prediction_date.dayofweek
        latest_record['Month'] = prediction_date.month
        latest_record['WeekNo'] = prediction_date.isocalendar().week
        latest_record['Year'] = prediction_date.year
        latest_record['Quarter'] = prediction_date.quarter
        
        # Update schedule-based features
        latest_record['IsSunday'] = 1 if prediction_date.dayofweek == 6 else 0
        latest_record['IsWeekend'] = 1 if prediction_date.dayofweek >= 5 else 0
        latest_record['IsMonday'] = 1 if prediction_date.dayofweek == 0 else 0
        latest_record['IsFriday'] = 1 if prediction_date.dayofweek == 4 else 0
        
        return latest_record
        
    except Exception as e:
        logger.error(f"❌ Error creating prediction features: {str(e)}")
        return None

def enhanced_predict_next_day(df, models, features, prediction_date=None):
    """
    Predict using enhanced models for next day
    Follows enterprise patterns from train_models2.py
    """
    try:
        if prediction_date is None:
            prediction_date = df['Date'].max() + timedelta(days=1)
        
        logger.info(f"🎯 Making enhanced prediction for {prediction_date.strftime('%Y-%m-%d')}")
        
        predictions = {}
        hours_predictions = {}
        
        for work_type in ['206', '213']:
            # Check if it's a working day for this punch code
            if not is_working_day_for_punch_code(prediction_date, work_type):
                predictions[work_type] = 0
                hours_predictions[work_type] = 0.0
                logger.info(f"📅 {work_type}: Non-working day - set to 0")
                continue
            
            # Check if model exists
            if work_type not in models:
                logger.warning(f"⚠️ No model found for {work_type}")
                predictions[work_type] = 0
                hours_predictions[work_type] = 0.0
                continue
            
            # Create prediction features
            pred_features = create_prediction_features(df, work_type, prediction_date)
            
            if pred_features is None:
                logger.warning(f"⚠️ Could not create features for {work_type}")
                predictions[work_type] = 0
                hours_predictions[work_type] = 0.0
                continue
            
            # Get selected features for this work type
            selected_features = features.get(work_type, [])
            
            if not selected_features:
                logger.warning(f"⚠️ No features defined for {work_type}")
                predictions[work_type] = 0
                hours_predictions[work_type] = 0.0
                continue
            
            # Filter to selected features
            available_features = [f for f in selected_features if f in pred_features.columns]
            
            if len(available_features) < len(selected_features) * 0.8:
                logger.warning(f"⚠️ Missing features for {work_type}: {len(available_features)}/{len(selected_features)}")
            
            # Make prediction
            try:
                X_pred = pred_features[available_features]
                
                # Handle missing values
                X_pred = X_pred.fillna(X_pred.median())
                
                # Make prediction
                model = models[work_type]
                pred_value = model.predict(X_pred)[0]
                
                # Ensure reasonable bounds
                pred_value = max(0, round(pred_value))
                
                predictions[work_type] = int(pred_value)
                hours_predictions[work_type] = float(pred_value * 8.0)  # Estimate hours
                
                logger.info(f"✅ {work_type}: Predicted {pred_value} workers")
                
            except Exception as e:
                logger.error(f"❌ Prediction error for {work_type}: {str(e)}")
                predictions[work_type] = 0
                hours_predictions[work_type] = 0.0
        
        return prediction_date, predictions, hours_predictions
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return prediction_date, {}, {}

def enhanced_predict_multiple_days(df, models, features, num_days=7):
    """
    Predict multiple days using enhanced models
    Enterprise-grade implementation
    """
    try:
        logger.info(f"📊 Making enhanced predictions for {num_days} days")
        
        multi_day_predictions = {}
        multi_day_hours = {}
        current_df = df.copy()
        
        latest_date = current_df['Date'].max()
        
        for i in range(num_days):
            prediction_date = latest_date + timedelta(days=i+1)
            
            # Make prediction for this day
            _, day_predictions, day_hours = enhanced_predict_next_day(
                current_df, models, features, prediction_date
            )
            
            multi_day_predictions[prediction_date] = day_predictions
            multi_day_hours[prediction_date] = day_hours
            
            # Add predictions to dataframe for next iteration
            new_rows = []
            for work_type, pred_value in day_predictions.items():
                new_row = {
                    'Date': prediction_date,
                    'WorkType': work_type,
                    'NoOfMan': pred_value,
                    'Hours': day_hours.get(work_type, 0.0),
                    'DayOfWeek': prediction_date.dayofweek,
                    'Month': prediction_date.month,
                    'Year': prediction_date.year,
                    'Quarter': prediction_date.quarter
                }
                new_rows.append(new_row)
            
            if new_rows:
                current_df = pd.concat([current_df, pd.DataFrame(new_rows)], ignore_index=True)
        
        logger.info(f"✅ Enhanced multi-day predictions complete")
        return multi_day_predictions, multi_day_hours
        
    except Exception as e:
        logger.error(f"❌ Error in multi-day prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}

def ensure_enhanced_data():
    """
    Ensure enhanced data and models are loaded
    Follows enterprise error handling patterns
    """
    # Load enhanced training data
    if st.session_state.enhanced_df is None:
        with st.spinner("🔗 Loading enhanced training data..."):
            st.session_state.enhanced_df = load_enhanced_training_data()
            
            if st.session_state.enhanced_df is not None:
                # Create enhanced features
                st.session_state.enhanced_df = create_enhanced_features(st.session_state.enhanced_df)
    
    # Load enhanced models
    if st.session_state.enhanced_models is None:
        with st.spinner("📂 Loading enhanced models..."):
            models, metadata, features = load_enhanced_models()
            
            if models is not None:
                st.session_state.enhanced_models = models
                st.session_state.enhanced_metadata = metadata
                st.session_state.enhanced_features = features
    
    # Check if we have everything needed
    if st.session_state.enhanced_df is None:
        st.error("❌ Could not load enhanced training data. Please check database connection.")
        return False
    
    if st.session_state.enhanced_models is None:
        st.error("❌ Could not load enhanced models. Please run train_models2.py first.")
        return False
    
    return True

def main():
    """
    Main enhanced predictions interface
    Enterprise-grade Streamlit application
    """
    st.header("🎯 Enhanced Predictions (Punch Codes 206 & 213)")
    
    st.info("""
    **Enhanced Prediction System**
    - Uses advanced models from `train_models2.py`
    - Specialized for punch codes 206 (6-day schedule) and 213 (5-day schedule)
    - Features advanced time series engineering and cross-validation
    - Enterprise-grade accuracy optimization
    """)
    
    # Ensure data and models are loaded
    if not ensure_enhanced_data():
        return
    
    # Display model information
    st.subheader("📊 Enhanced Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Available Models:**")
        for work_type in st.session_state.enhanced_models.keys():
            metadata = st.session_state.enhanced_metadata.get(work_type, {})
            mae = metadata.get('final_mae', 'N/A')
            r2 = metadata.get('final_r2', 'N/A')
            st.write(f"- Punch Code {work_type}: MAE={mae:.3f}, R²={r2:.3f}" if isinstance(mae, float) else f"- Punch Code {work_type}")
    
    with col2:
        st.write("**Data Summary:**")
        if st.session_state.enhanced_df is not None:
            for work_type in ['206', '213']:
                work_data = st.session_state.enhanced_df[st.session_state.enhanced_df['WorkType'] == work_type]
                st.write(f"- Punch Code {work_type}: {len(work_data)} records")
    
    # Prediction interface
    st.subheader("🔮 Make Enhanced Predictions")
    
    prediction_type = st.radio(
        "Select Prediction Type:",
        ["Next Day", "Multiple Days"],
        horizontal=True
    )
    
    if prediction_type == "Next Day":
        # Next day prediction
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Predict Next Day", type="primary"):
                with st.spinner("Making enhanced prediction..."):
                    latest_date = st.session_state.enhanced_df['Date'].max()
                    next_date, predictions, hours_pred = enhanced_predict_next_day(
                        st.session_state.enhanced_df,
                        st.session_state.enhanced_models,
                        st.session_state.enhanced_features
                    )
                    
                    st.session_state.enhanced_predictions = {
                        'date': next_date,
                        'predictions': predictions,
                        'hours': hours_pred
                    }
        
        # Display next day results
        if st.session_state.enhanced_predictions:
            pred_data = st.session_state.enhanced_predictions
            
            st.success(f"📅 Predictions for {pred_data['date'].strftime('%Y-%m-%d')} ({pred_data['date'].strftime('%A')})")
            
            # Create results table
            results = []
            for work_type in ['206', '213']:
                workers = pred_data['predictions'].get(work_type, 0)
                hours = pred_data['hours'].get(work_type, 0.0)
                
                # Check working day status
                is_working = is_working_day_for_punch_code(pred_data['date'], work_type)
                status = "✅ Working" if is_working else "📅 Non-working"
                
                results.append({
                    'Punch Code': work_type,
                    'Predicted Workers': workers,
                    'Predicted Hours': f"{hours:.1f}",
                    'Status': status
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
    
    else:
        # Multiple days prediction
        col1, col2 = st.columns(2)
        
        with col1:
            num_days = st.slider("Number of days to predict:", 1, 14, 7)
        
        with col2:
            if st.button("📊 Predict Multiple Days", type="primary"):
                with st.spinner(f"Making {num_days}-day enhanced predictions..."):
                    multi_predictions, multi_hours = enhanced_predict_multiple_days(
                        st.session_state.enhanced_df,
                        st.session_state.enhanced_models,
                        st.session_state.enhanced_features,
                        num_days
                    )
                    
                    if multi_predictions:
                        # Create results dataframe
                        results = []
                        for date, predictions in multi_predictions.items():
                            for work_type in ['206', '213']:
                                workers = predictions.get(work_type, 0)
                                hours = multi_hours[date].get(work_type, 0.0)
                                
                                is_working_result = is_working_day_for_punch_code(date, work_type)
                                # Convert to boolean and string status
                                is_working_bool = isinstance(is_working_result, bool) and is_working_result
                                working_status = "Yes" if is_working_bool else "No"

                                results.append({
                                    'Date': date.strftime('%Y-%m-%d'),
                                    'Day': date.strftime('%A'),
                                    'Punch Code': work_type,
                                    'Predicted Workers': workers,
                                    'Predicted Hours': round(hours, 1),
                                    'Working Day': working_status  # ✅ Clean string value
                                })
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.success(f"📊 {num_days}-Day Enhanced Predictions Complete")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        for work_type in ['206', '213']:
                            work_results = results_df[results_df['Punch Code'] == work_type]
                            
                            fig.add_trace(go.Scatter(
                                x=work_results['Date'],
                                y=work_results['Predicted Workers'],
                                mode='lines+markers',
                                name=f'Punch Code {work_type}',
                                line=dict(width=3),
                                marker=dict(size=8)
                            ))
                        
                        fig.update_layout(
                            title="Enhanced Multi-Day Workforce Predictions",
                            xaxis_title="Date",
                            yaxis_title="Predicted Workers",
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download option
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Enhanced Predictions (CSV)",
                            data=csv_data,
                            file_name=f"enhanced_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
    
    # Performance metrics section
    if st.session_state.enhanced_metadata:
        st.subheader("📈 Enhanced Model Performance")
        
        performance_data = []
        for work_type, metadata in st.session_state.enhanced_metadata.items():
            if metadata:
                performance_data.append({
                    'Punch Code': work_type,
                    'MAE': f"{metadata.get('final_mae', 0):.3f}",
                    'R²': f"{metadata.get('final_r2', 0):.3f}",
                    'MAPE': f"{metadata.get('mape', 0):.2f}%",
                    'CV MAE': f"{metadata.get('cv_mae', 0):.3f}",
                    'Training Records': metadata.get('training_records', 0),
                    'Features': len(metadata.get('features', []))
                })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)

if __name__ == "__main__":
    main()