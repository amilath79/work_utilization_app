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
from utils.state_manager import StateManager

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_data
from utils.feature_engineering import engineer_features, create_lag_features
from config import DATA_DIR

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

# Check if we have data
def ensure_data():
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
    if st.session_state.processed_df is None:
        with st.spinner("Processing data..."):
            st.session_state.processed_df = engineer_features(st.session_state.df)
    
    if st.session_state.ts_data is None:
        with st.spinner("Creating time series features..."):
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
        work_type_data.head(11),
        x='WorkType',
        y='NoOfMan',
        title='Top 11 Punch Codes by Total Workers',
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
    top_work_types = work_type_data.head(11)['WorkType'].tolist()
    selected_work_types = st.multiselect(
        "Select Work Types to Visualize",
        options=work_type_data['WorkType'].unique(),
        default=top_work_types[:5]
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
    
    # Get data
    df = st.session_state.df
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Data Summary", "Time Analysis", "Work Type (Punch Code) Analysis"])
    
    with tab1:
        display_data_summary(df)
    
    with tab2:
        display_time_analysis(df)
    
    with tab3:
        display_work_type_analysis(df)

# Run the main function
if __name__ == "__main__":
    main()
    StateManager.initialize()