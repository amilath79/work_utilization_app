"""
Home page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import logging
import os
import sys
from datetime import datetime
import traceback
from config import ENTERPRISE_CONFIG, enterprise_logger

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.feature_engineering import EnhancedFeatureTransformer
from utils.data_loader import load_data, load_combined_models
from utils.sql_data_connector import extract_sql_data
from utils.feature_engineering import EnhancedFeatureTransformer
from utils.state_manager import StateManager
from config import DATA_DIR, MODELS_DIR, LOGO_PATH, APP_TITLE, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, APP_ICON

# Configure page - SINGLE CONFIG ONLY
if ENTERPRISE_CONFIG.enterprise_mode:
    st.set_page_config(
        page_title=f"Enterprise - {APP_TITLE}",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
else:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Configure logger
logger = logging.getLogger(__name__)

def load_data_from_database():
    """
    Load data from the WorkUtilizationData database table using configuration from config.py
    Returns DataFrame on success, None on failure
    """
    try:
        # Create SQL query for WorkUtilizationData table
        sql_query = """
        SELECT Date, PunchCode as WorkType, Hours, NoOfMan, SystemHours, NoRows as Quantity, SystemKPI 
        FROM WorkUtilizationData 
        WHERE PunchCode IN ('202', '203', '206', '209', '210', '211', '213', '214', '215', '217') 
        AND Hours > 0 
        AND NoOfMan > 0 
        AND SystemHours > 0 
        AND NoRows > 0
        ORDER BY Date
        """
        
        # Show connecting message
        with st.spinner(f"Connecting to database {SQL_DATABASE} on {SQL_SERVER}..."):
            # Use the extract_sql_data function from sql_data_connector.py
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
                
                logger.info(f"Successfully loaded {len(df)} records from database")
                return df
            else:
                logger.warning("No data returned from database query")
                return None
    except Exception as e:
        logger.error(f"Error loading data from database: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():

    # Simple enterprise page configuration
    if ENTERPRISE_CONFIG.enterprise_mode:

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏢 Enterprise Mode")
        st.sidebar.markdown(f"🔒 Environment: {ENTERPRISE_CONFIG.environment.value}")

        
    # Display logo if available
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=50)
    
    st.title("Work Utilization Prediction")
    
    st.write("""
    Welcome to the Work Utilization Prediction application! This tool helps you predict
    workforce requirements based on historical data using machine learning.
    
    Use the sidebar to navigate between different pages.
    """)
    
    # Data Loading Section
    st.header("↗️ Data Management")
    
    # Check if data is already loaded
    if 'df' in st.session_state and st.session_state.df is not None:
        st.success(f"✅ Data loaded successfully with {len(st.session_state.df):,} records from {st.session_state.df['Date'].min().strftime('%Y-%m-%d')} to {st.session_state.df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Option to clear and reload data
        if st.button("Clear Data and Reload"):
            st.session_state.df = None
            st.session_state.processed_df = None
            st.session_state.ts_data = None
            st.rerun()
    else:
        # First, try to load data from database
        db_data = load_data_from_database()
        
        if db_data is not None:
            # Successfully loaded from database
            st.session_state.df = db_data
            st.success(f"✅ Data loaded from database with {len(db_data):,} records from {db_data['Date'].min().strftime('%Y-%m-%d')} to {db_data['Date'].max().strftime('%Y-%m-%d')}")
            st.rerun()
        else:
            # Database connection failed, show Excel options
            st.warning("⚠️ Could not connect to database. Please upload data from Excel file instead.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload Excel File", 
                    type=["xlsx", "xls"],
                    help="Upload your Work Utilization Excel file"
                )
            
            with col2:
                use_sample_data = st.checkbox(
                    "Use Sample Data", 
                    value=False,
                    help="Use sample data if you don't have your own file"
                )
            
            if uploaded_file is not None:
                # Use uploaded file
                with st.spinner("Loading data from uploaded file..."):
                    st.session_state.df = load_data(uploaded_file)
                    st.success(f"✅ Data loaded successfully with {len(st.session_state.df):,} records")
                    st.rerun()
                    
            elif use_sample_data:
                # Use sample data
                sample_path = os.path.join(DATA_DIR, "sample_work_utilization.xlsx")
                
                if os.path.exists(sample_path):
                    with st.spinner("Loading sample data..."):
                        st.session_state.df = load_data(sample_path)
                        st.success(f"✅ Sample data loaded successfully with {len(st.session_state.df):,} records")
                        st.rerun()
                else:
                    st.warning("Sample data file not found. Please upload your own data.")
    
    # Process data if not already done
    if 'df' in st.session_state and st.session_state.df is not None:
        if 'processed_df' not in st.session_state or st.session_state.processed_df is None:
            with st.spinner("Processing data..."):
                # Initialize and use the transformer
                feature_transformer = EnhancedFeatureTransformer()
                
                # Fit and transform the data
                feature_transformer.fit(st.session_state.df)
                st.session_state.processed_df = feature_transformer.transform(st.session_state.df)
                st.session_state.ts_data = st.session_state.processed_df  # Already includes lag features
    
    # Models Section
    st.header("🤖 Models")
    
    # Check if models are already loaded
    if 'models' in st.session_state and st.session_state.models is not None:
        st.success(f"✅ Models loaded successfully. {len(st.session_state.models)} work type models available.")
        
        # Display model info
        if st.checkbox("Show Model Details"):
            if 'metrics' in st.session_state and st.session_state.metrics is not None:
                metrics_df = pd.DataFrame([
                    {
                        'Work Type': wt,
                        'MAE': m.get('MAE', '-'),
                        'RMSE': m.get('RMSE', '-'),
                        'R²': m.get('R²', '-')
                    }
                    for wt, m in st.session_state.metrics.items()
                ]).sort_values('MAE')
                
                st.dataframe(metrics_df, use_container_width=True)
    else:
        # Try to load models
        try:
            with st.spinner("Loading all models..."):
                models, feature_importances, metrics = load_combined_models()
                
                if models:
                    st.session_state.models = models
                    st.session_state.feature_importances = feature_importances
                    st.session_state.metrics = metrics
                    st.success(f"✅ Models loaded successfully. {len(models)} work type models available.")
                else:
                    st.warning("No trained models found. You need to train models before making predictions.")
                    
                    if st.button("Train Models"):
                        st.info("Training models... This may take several minutes.")
                        st.write("Please run 'python train_models.py' from the command line to train the models.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
    
    # Quick Start Guide
    with st.expander("📚 Quick Start Guide", expanded=True):
        st.write("""
        ### How to use this application
        
        1. **Load Data**: By default, data is loaded from the database. If that fails, you can upload an Excel file.
        2. **Explore Data**: Go to the Data Overview page to explore trends and patterns
        3. **Generate Predictions**: Use the Predictions page to forecast workforce requirements
        4. **Analyze Models**: Check model performance and feature importance on the Model Analysis page
        5. **Manage Non-Working Days**: View holidays and non-working days on the dedicated page
        
        ### Data Format Requirements
        
        Your database table or Excel file should contain the following columns:
        - **Date**: Date of the utilization record
        - **WorkType** or **PunchCode**: Type of work being performed
        - **NoOfMan**: Number of workers utilized
        - **Hours**: Optional - Hours worked
        - **SystemHours**: Optional - System hours recorded
        - **Quantity**: Optional - Quantity of work processed
        - **ResourceKPI**: Optional - Resource KPI
        - **SystemKPI**: Optional - System KPI
        """)
    
    # System Status
    st.sidebar.header("System Status")
    
    # Check if data is loaded
    if 'df' in st.session_state and st.session_state.df is not None:
        st.sidebar.success("✅ Data loaded")
    else:
        st.sidebar.warning("❌ No data loaded")
    
    # Check if models are loaded
    if 'models' in st.session_state and st.session_state.models is not None:
        st.sidebar.success("✅ Models loaded")
    else:
        st.sidebar.warning("❌ No models loaded")
    
    # Display current date
    st.sidebar.write(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
    StateManager.initialize()