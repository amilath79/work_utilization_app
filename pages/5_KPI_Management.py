"""
KPI Management page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import logging
import traceback
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_manager import StateManager
from utils.kpi_manager import load_kpi_data, save_kpi_data, initialize_kpi_dataframe, save_financial_year

# Configure page
st.set_page_config(
    page_title="KPI Management",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

def ensure_dataframe(data):
    """
    Ensure data is a valid pandas DataFrame regardless of input type
    """
    try:
        if data is None:
            return pd.DataFrame({'Date': []})
        elif isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            if not data:  # Empty list
                return pd.DataFrame({'Date': []})
            elif isinstance(data[0], dict):  # List of dicts
                return pd.DataFrame(data)
            else:  # List of values
                return pd.DataFrame({'Value': data})
        else:
            # Last resort - just convert to string
            return pd.DataFrame({'Value': [str(data)]})
    except Exception as e:
        logger.error(f"Error ensuring DataFrame: {str(e)}")
        # Return an empty DataFrame as fallback
        return pd.DataFrame({'Date': []})

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
    
    # Initialize or get dataframe and attempt to auto-load data
    if st.session_state.kpi_df is None:
        try:
            # First try to load data from database
            loaded_df = load_kpi_data(from_date, to_date, date_range_type.upper())
            
            # Use loaded data if available, otherwise initialize empty
            if loaded_df is not None and not loaded_df.empty:
                st.session_state.kpi_df = ensure_dataframe(loaded_df)
            else:
                st.session_state.kpi_df = ensure_dataframe(initialize_kpi_dataframe(from_date, to_date, date_range_type))
        except Exception as e:
            logger.error(f"Error auto-loading KPI data: {str(e)}")
            # Fall back to empty dataframe on error
            st.session_state.kpi_df = ensure_dataframe(initialize_kpi_dataframe(from_date, to_date, date_range_type))
    
    # Check if we're in monthly view for financial year
    is_financial_year_view = (date_range_type == "Monthly" and 
                             from_date.month == 5 and from_date.day == 1 and
                             to_date.month == 4 and to_date.day == 30 and
                             to_date.year == from_date.year + 1)
                      
    if is_financial_year_view:
        st.info(f"🗓️ Financial Year View: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
    
    st.subheader(f"Manage {date_range_type} KPIs")
    
    # Always ensure kpi_df is a DataFrame with the robust function
    kpi_df = ensure_dataframe(st.session_state.kpi_df)
    
    # Store the original data in session state for later reference
    st.session_state.original_kpi_df = kpi_df.copy()
    
    # Use the data editor - now without saving the return value
    st.data_editor(
        kpi_df,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        key="kpi_data_editor",
        column_config={
            "Date": st.column_config.TextColumn("Date", disabled=True)
        }
    )
    
    # Save buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Save button
        if st.button("Save KPI Data", type="secondary"):
            with st.spinner("Saving KPI data..."):
                try:
                    # Get current user from system
                    username = os.environ.get('USERNAME', 'Unknown')
                    
                    # Get the original dataframe
                    save_df = st.session_state.original_kpi_df.copy()
                    
                    # Check if we have edits and apply them if available
                    if 'kpi_data_editor' in st.session_state:
                        edit_data = st.session_state.kpi_data_editor
                        
                        # Apply edits from the editor if they exist
                        if 'edited_rows' in edit_data and edit_data['edited_rows']:
                            for row_idx, edits in edit_data['edited_rows'].items():
                                row_idx = int(row_idx)  # Convert string index to integer
                                for col, value in edits.items():
                                    save_df.at[row_idx, col] = value
                        
                        # No need to handle added_rows or deleted_rows as they're not enabled in this interface
                    
                    # Save using the updated dataframe
                    success = save_kpi_data(
                        save_df,
                        from_date,
                        to_date,
                        username,
                        date_range_type.upper()
                    )
                    
                    if success:
                        # Update session state with the saved data
                        st.session_state.kpi_df = save_df
                        st.success("KPI data saved successfully!")
                        st.rerun()  # Refresh the page to show updated data
                    else:
                        st.error("Error saving KPI data. Please check the application logs for details.")
                except Exception as e:
                    logger.error(f"Error in save button handler: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error(f"An unexpected error occurred: {str(e)}")

    with col2:
        # Financial year button (only show in financial year view)
        if is_financial_year_view:
            if st.button("Register as Financial Year", type="primary"):
                with st.spinner("Registering financial year..."):
                    try:
                        # Get current user from system
                        username = os.environ.get('USERNAME', 'Unknown')
                        
                        # Save to KPIFinancialYears table
                        success = save_financial_year(
                            from_date,
                            to_date,
                            username
                        )
                        
                        if success:
                            st.success(f"Financial year {from_date.year}-{to_date.year} registered successfully!")
                        else:
                            st.error("Error registering financial year. Please check the application logs for details.")
                    except Exception as e:
                        logger.error(f"Error in financial year handler: {str(e)}")
                        logger.error(traceback.format_exc())
                        st.error(f"An unexpected error occurred: {str(e)}")
                        
if __name__ == "__main__":
    main()
    StateManager.initialize()