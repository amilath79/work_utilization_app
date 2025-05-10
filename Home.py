"""
Main application entry point for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
os.environ["STREAMLIT_SERVER_WATCH_CHANGES"] = "false"
import sys
from datetime import datetime, timedelta

# Import from other modules
from utils.data_loader import load_data, load_models
from utils.feature_engineering import engineer_features, create_lag_features



from config import (
    APP_TITLE, 
    APP_ICON, 
    DEFAULT_LAYOUT,
    THEME_COLOR,
    LOGO_PATH,
    MODELS_DIR,
    DATA_DIR,
    LOGS_DIR
)


# Page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=DEFAULT_LAYOUT,
    initial_sidebar_state="expanded",
)

# Check if the theme is dark
is_dark_theme = st.get_option("theme.base") == "dark"

# Use different colors based on theme
primary_color = "#1E88E5" if not is_dark_theme else "#4DA6FF"
background_color = "#f8f9fa" if not is_dark_theme else "#262730"
text_color = "#212529" if not is_dark_theme else "#FAFAFA"

# Then use these variables in your app
st.markdown(f"""
<style>
.custom-header {{
    color: {text_color};
    background-color: {background_color};
}}
</style>
""", unsafe_allow_html=True)

# Configure logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Apply custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    /* Theme-independent styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Light theme styles */
    .light-mode {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Dark theme styles */
    .dark-mode {
        background-color: #262730;
        color: #ffffff;
    }
    
    /* Use theme-sensitive color variables that Streamlit provides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--background-secondary);
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: var(--text-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: var(--background-color);
    }
    </style>
    """, unsafe_allow_html=True)

def add_theme_toggle():
    """Add theme toggle to sidebar"""
    with st.sidebar:
        st.markdown("---")
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=1
        )
        if theme == "Dark":
            st.markdown("""
            <script>
                const setDarkMode = () => {
                    document.documentElement.classList.add('dark-theme');
                    localStorage.setItem('theme', 'dark');
                };
                setDarkMode();
            </script>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <script>
                const setLightMode = () => {
                    document.documentElement.classList.remove('dark-theme');
                    localStorage.setItem('theme', 'light');
                };
                setLightMode();
            </script>
            """, unsafe_allow_html=True)

# Create session state for data persistence
def init_session_state():
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




# Display app header
def display_header():
    col1, col2 = st.columns([1, 5])
    
    # Display logo if available
    try:
        with col1:
            if os.path.exists(LOGO_PATH):
                st.image(LOGO_PATH, width=80)
            else:
                st.write("📊")
    except:
        st.write("📊")
        
    with col2:
        st.title(APP_TITLE)
    
    st.markdown("---")

# Sidebar components
def sidebar_components():
    with st.sidebar:
        # Show logo if available
        try:
            if os.path.exists(LOGO_PATH):
                st.image(LOGO_PATH, width=100)
            else:
                st.title("📊 WorkForce AI")
        except:
            st.title("📊 WorkForce AI")
        
        st.markdown("---")
        
        # Data Input section - Moved from page-specific navigation
        st.header("Data Input")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Excel File", 
            type=["xlsx", "xls"],
            help="Upload your Work Utilization Excel file"
        )
        
        # Sample data option
        use_sample_data = st.checkbox(
            "Use Sample Data", 
            value=False,
            help="Use sample data if you don't have your own file"
        )
        
        st.markdown("---")
        
        # App information
        st.info(
            f"Workforce Prediction\n\n"
            f"Version: 1.1.5\n"
            f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        # Add theme toggle
        add_theme_toggle()
        
        # Return data options
        return uploaded_file, use_sample_data

def load_work_types():
    try:
        # Path to the Excel file
        excel_path = os.path.join(DATA_DIR, 'WorkTypes.xlsx')
        
        if os.path.exists(excel_path):
            # Load work types from Excel file once at startup
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

# Load and prepare data
def load_and_prepare_data(uploaded_file, use_sample_data):
    try:
        if uploaded_file is not None:
            # Use uploaded file
            logger.info(f"Loading data from uploaded file: {uploaded_file.name}")
            st.session_state.df = load_data(uploaded_file)
            return True
        elif use_sample_data:
            # Use sample data
            logger.info("Loading sample data")
            sample_path = os.path.join(DATA_DIR, "sample_work_utilization.xlsx")
            
            if os.path.exists(sample_path):
                st.session_state.df = load_data(sample_path)
                return True
            else:
                st.warning("Sample data file not found. Please upload your own data.")
                return False
        else:
            if st.session_state.df is None:
                st.info("Please upload an Excel file or use sample data to get started.")
                return False
            return True
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        st.error("Please ensure your file has the correct format with columns: Date, WorkType, NoOfMan, etc.")
        return False

# Load models if available
def load_prediction_models():
    try:
        if st.session_state.models is None:
            models, feature_importances, metrics = load_models()
            
            if models:
                st.session_state.models = models
                st.session_state.feature_importances = feature_importances
                st.session_state.metrics = metrics
                logger.info("Models loaded successfully")
                return True
            else:
                if not os.path.exists(os.path.join(MODELS_DIR, 'work_utilization_models.pkl')):
                    st.info("No trained models found. Please run the notebook first to train models.")
                else:
                    st.warning("Error loading models. Please check the logs for details.")
                return False
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        st.warning(f"Error loading models: {str(e)}")
        return False

# Main function to run the Streamlit app
def main():
    try:
        # Apply custom CSS
        apply_custom_css()
        
        # Initialize session state
        init_session_state()
        
        # Display app header
        display_header()
        
        # Sidebar components - just the data loading controls
        uploaded_file, use_sample_data = sidebar_components()
        
        # Load and prepare data
        data_loaded = load_and_prepare_data(uploaded_file, use_sample_data)
        
        # Process the data if it's loaded
        if data_loaded:
            # Ensure feature engineering is done
            if st.session_state.processed_df is None and st.session_state.df is not None:
                with st.spinner("Processing data..."):
                    st.session_state.processed_df = engineer_features(st.session_state.df)
            
            if st.session_state.ts_data is None and st.session_state.processed_df is not None:
                with st.spinner("Creating time series features..."):
                    st.session_state.ts_data = create_lag_features(st.session_state.processed_df)
            
            # Load models
            load_prediction_models()
            
            # Main page content - this is the home/welcome page
            st.subheader("Welcome to the Workforce Prediction App")
            st.write("""
            This application helps you predict workforce requirements based on historical data.
            
            Navigation:
            - **Data Overview**: Explore your data and view statistics
            - **Predictions**: Generate workforce predictions for future dates
            - **Model Analysis**: Analyze model performance and feature importance
            
            Your data has been loaded successfully. Use the navigation in the sidebar to explore the app.
            """)
            
            # Display a quick summary of the data
            if st.session_state.df is not None:
                st.subheader("Data Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", f"{len(st.session_state.df):,}")
                with col2:
                    st.metric("Date Range", f"{st.session_state.df['Date'].min().strftime('%Y-%m-%d')} to {st.session_state.df['Date'].max().strftime('%Y-%m-%d')}")
                with col3:
                    st.metric("Work Types (Punch Codes)", f"{st.session_state.df['WorkType'].nunique():,}")
                
                st.markdown("---")
                st.info("**Tip**: Use the **Data Overview** page to explore your data in more detail.")
        else:
            # Show welcome message if no data is loaded
            st.subheader("Welcome to the Workforce Prediction App")
            st.write("""
            This application helps you predict workforce requirements based on historical data.
            
            To get started:
            1. Upload your Excel file containing work utilization data, or use the sample data
            2. Explore your data in the Data Overview page
            3. Generate predictions in the Predictions page
            4. Analyze model performance in the Model Analysis page
            """)
            
            st.info("Please upload your data or select 'Use Sample Data' in the sidebar to begin.")
            
    except Exception as e:
        logger.error(f"Unexpected error in main app: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please check the logs for more details or contact support.")

if __name__ == "__main__":
    main()