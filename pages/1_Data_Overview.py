"""
Data Overview page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, timedelta
import traceback
import pyodbc
from utils.state_manager import StateManager

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_data
from utils.sql_data_connector import extract_sql_data
from utils.feature_engineering import engineer_features, create_lag_features
from config import DATA_DIR, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION

# Configure page
st.set_page_config(
    page_title="Data Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
import logging
logger = logging.getLogger(__name__)

def load_workutilizationdata():
    """
    Load data from the WorkUtilizationData table
    """
    try:
        # Create SQL query for all PunchCodes
        sql_query = """
        SELECT Date, PunchCode as WorkType, Hours, NoOfMan, SystemHours, Quantity, ResourceKPI, SystemKPI 
        FROM WorkUtilizationData
        WHERE PunchCode IN (215, 209, 213, 211, 214, 202, 203, 206, 208, 210, 217)
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

# Check if we have data and load if needed
def ensure_data():
    # Initialize session state variables if not present
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'ts_data' not in st.session_state:
        st.session_state.ts_data = None
    
    # First try to load from database
    if st.session_state.df is None:
        st.session_state.df = load_workutilizationdata()
    
    # If database load failed, offer Excel options
    if st.session_state.df is None:
        st.error("Could not load data from database. Please upload Excel file instead.")
        
        uploaded_file = st.file_uploader(
            "Upload Work Utilization Excel File", 
            type=["xlsx", "xls"],
            help="Upload Excel file with work utilization data (Hours, NoOfMan, SystemHours, Quantity, etc.)"
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
        
        # Check if we have data after trying upload options
        if st.session_state.df is None:
            st.warning("No data available. Please upload a file or connect to the database.")
            return False
    
    # Process data if available
    if st.session_state.df is not None and st.session_state.processed_df is None:
        with st.spinner("Processing data..."):
            st.session_state.processed_df = engineer_features(st.session_state.df)
            st.session_state.ts_data = create_lag_features(st.session_state.processed_df)
    
    return True

def display_data_summary(df):
    """Display summary statistics and information about the dataset"""
    st.subheader("Data Summary")
    
    # Basic information
    col1, col2, col3= st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    with col3:
        st.metric("Work Types", f"{df['WorkType'].nunique():,}")
    
    # Display sample of the data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Data types and missing values
    st.subheader("Data Information")
    
    # Create a DataFrame with column info
    col_info = []
    for col in df.columns:
        col_info.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Missing": df[col].isna().sum(),
            "Missing %": round(df[col].isna().sum() / len(df) * 100, 2),
            "Unique Values": df[col].nunique()
        })
    
    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df)

def display_time_analysis(df):
    """Display time-based analysis of the data"""
    st.subheader("Time-Based Analysis")
    
    # Aggregate by date
    daily_data = df.groupby('Date')['NoOfMan'].sum().reset_index()
    daily_data['Day of Week'] = daily_data['Date'].dt.day_name()
    daily_data['Month'] = daily_data['Date'].dt.month_name()
    daily_data['Year'] = daily_data['Date'].dt.year
    
    # Display time series plot
    st.write("### Daily Worker Count Over Time")
    
    fig = px.line(
        daily_data, 
        x='Date', 
        y='NoOfMan',
        title='Total Workers Over Time',
        labels={'NoOfMan': 'Number of Workers', 'Date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Workers',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    st.write("### Workers by Day of Week")
    
    dow_data = daily_data.groupby('Day of Week')['NoOfMan'].mean().reset_index()
    # Ensure days of week are in correct order
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_data['Day of Week'] = pd.Categorical(dow_data['Day of Week'], categories=dow_order, ordered=True)
    dow_data = dow_data.sort_values('Day of Week')
    
    fig = px.bar(
        dow_data,
        x='Day of Week',
        y='NoOfMan',
        title='Average Workers by Day of Week',
        color='NoOfMan',
        labels={'NoOfMan': 'Average Workers', 'Day of Week': 'Day of Week'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly analysis
    st.write("### Workers by Month")
    
    month_data = daily_data.groupby(['Year', 'Month'])['NoOfMan'].mean().reset_index()
    # Ensure months are in correct order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_data['Month'] = pd.Categorical(month_data['Month'], categories=month_order, ordered=True)
    month_data = month_data.sort_values(['Year', 'Month'])
    
    fig = px.line(
        month_data,
        x='Month',
        y='NoOfMan',
        color='Year',
        title='Average Workers by Month',
        labels={'NoOfMan': 'Average Workers', 'Month': 'Month', 'Year': 'Year'},
        markers=True
    )
    
    # Explicitly set the x-axis category order to ensure January to December sequence
    fig.update_layout(
        xaxis=dict(
            categoryorder='array',
            categoryarray=month_order
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_work_type_analysis(df):
    """Display work type analysis"""
    st.subheader("Work Type (Punch Code) Analysis")
    
    # Aggregate by work type
    work_type_data = df.groupby('WorkType')['NoOfMan'].sum().reset_index()
    work_type_data = work_type_data.sort_values('NoOfMan', ascending=False)
    
    # Top work types
    st.write("### Top Punch Code by Total Workers")
    
    fig = px.bar(
        work_type_data,
        x='WorkType',
        y='NoOfMan',
        title='Punch Codes by Total Workers',
        color='NoOfMan',
        labels={'NoOfMan': 'Total Workers', 'WorkType': 'Work Type'}
    )
    
    fig.update_layout(
        xaxis_title='Punch Code',
        yaxis_title='Total Workers',
        xaxis={'categoryorder': 'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Work type trends over time
    st.write("### Punch Code Trends")
    
    # Allow user to select work types to visualize
    top_work_types = work_type_data.head(5)['WorkType'].tolist()
    selected_work_types = st.multiselect(
        "Select Work Types to Visualize",
        options=work_type_data['WorkType'].unique(),
        default=top_work_types
    )
    
    if selected_work_types:
        # Filter data for selected work types
        filtered_data = df[df['WorkType'].isin(selected_work_types)]
        
        # Aggregate by date and work type
        trend_data = filtered_data.groupby(['Date', 'WorkType'])['NoOfMan'].sum().reset_index()
        
        fig = px.line(
            trend_data,
            x='Date',
            y='NoOfMan',
            color='WorkType',
            title='Workers Over Time by Work Type',
            labels={'NoOfMan': 'Number of Workers', 'Date': 'Date', 'WorkType': 'Work Type'}
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Workers',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one work type to visualize trends.")

def main():
    st.header("Data Overview")
    
    # Check if data is loaded
    if not ensure_data():
        return
    
    # Success message when data is loaded
    if st.session_state.df is not None:
        st.success(f"✅ Data loaded with {len(st.session_state.df):,} records")
        
        # Create analysis tabs
        analysis_tabs = st.tabs(["Data Summary", "Time Analysis", "Work Type (Punch Code) Analysis"])
        
        with analysis_tabs[0]:
            display_data_summary(st.session_state.df)
        
        with analysis_tabs[1]:
            display_time_analysis(st.session_state.df)
        
        with analysis_tabs[2]:
            display_work_type_analysis(st.session_state.df)

# Run the main function
if __name__ == "__main__":
    main()
    StateManager.initialize()