"""
KPI Management page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_manager import StateManager
from utils.sql_data_connector import SQLDataConnector
from utils.kpi_manager import load_kpi_data, save_kpi_data, initialize_kpi_dataframe, load_punch_codes
from config import DATA_DIR, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION

# Configure page
st.set_page_config(
    page_title="KPI Management",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

def main():
    st.header("📈 KPI Management")
    
    st.info("""
    This page allows you to manage KPIs (Key Performance Indicators) for different punch codes.
    You can set and edit KPI values for daily, weekly, or monthly periods.
    """)
    
    # Initialize session state for data persistence
    if 'kpi_df' not in st.session_state:
        st.session_state.kpi_df = None
    if 'date_range_type' not in st.session_state:
        st.session_state.date_range_type = "Daily"
    
    # Date range selection
    st.subheader("Select Date Range")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Date range type selection
        date_range_type = st.radio(
            "Select Date Range Type",
            ["Daily", "Weekly", "Monthly"],
            index=0,
            horizontal=True,
            key="date_range_type_radio"
        )
        # Update session state if changed
        st.session_state.date_range_type = date_range_type
    
    with col2:
        # Start date
        from_date = st.date_input(
            "From",
            value=datetime.now().date(),
            help="Select start date"
        )
    
    with col3:
        # End date
        to_date = st.date_input(
            "To",
            value=datetime.now().date() + timedelta(days=7),
            help="Select end date"
        )
    
    # Check if dates or period type changed - if so, reset the dataframe
    date_key = f"{from_date}-{to_date}-{date_range_type}"
    if 'last_date_key' not in st.session_state or st.session_state.last_date_key != date_key:
        st.session_state.kpi_df = None
        st.session_state.last_date_key = date_key
    
    # Initialize or get dataframe
    if st.session_state.kpi_df is None:
        kpi_df = initialize_kpi_dataframe(from_date, to_date, date_range_type)
        st.session_state.kpi_df = kpi_df
    else:
        kpi_df = st.session_state.kpi_df
    
    # Load button
    if st.button("Load KPI Data", type="primary"):
        with st.spinner(f"Loading KPI data for {date_range_type.lower()} view..."):
            try:
                # Call utility function to load data
                loaded_df = load_kpi_data(from_date, to_date, date_range_type.upper())
                
                if not loaded_df.empty:
                    st.session_state.kpi_df = loaded_df
                    kpi_df = loaded_df
                    st.success(f"KPI data loaded for {date_range_type.lower()} view")
                else:
                    st.info("No existing KPI data found. You can enter new values below.")
            except Exception as e:
                logger.error(f"Error loading KPI data: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"An error occurred while loading KPI data: {str(e)}")
    
    # KPI data editor
    st.subheader(f"Manage {date_range_type} KPIs")
    
    # Use Streamlit's data editor to make the table editable
    edited_df = st.data_editor(
        kpi_df,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        key="kpi_data_editor"
    )
    
    # Immediately update session state with edited values to preserve changes
    st.session_state.kpi_df = edited_df
    
    # Save button
    if st.button("Save KPI Data", type="secondary"):
        with st.spinner("Saving KPI data..."):
            try:
                # Get current user from system
                username = os.environ.get('USERNAME', 'Unknown')
                
                # Important: Use the edited dataframe from session state
                success = save_kpi_data(
                    st.session_state.kpi_df,  # Use session state data
                    from_date,
                    to_date,
                    username,
                    date_range_type.upper()
                )
                
                if success:
                    st.success("KPI data saved successfully!")
                else:
                    st.error("Error saving KPI data. Please check the application logs for details.")
            except Exception as e:
                logger.error(f"Error in save button handler: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
    StateManager.initialize()