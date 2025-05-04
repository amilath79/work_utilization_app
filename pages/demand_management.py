"""
Demand Management page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Configure logger
logger = logging.getLogger(__name__)

def render(df, models=None):
    """
    Render the Demand Management page
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataset
    models : dict, optional
        Dictionary of trained models for each WorkType
    """
    st.header("Demand Management")
    
    # Get the latest date from the dataset
    latest_date = df['Date'].max().date()
    
    # Create tabs for different operations
    tab1, tab2 = st.tabs(["Update Demand", "View Demand History"])
    
    with tab1:
        st.subheader("Update Demand Parameters")
        
        # Date selector (next working day)
        col1, col2 = st.columns(2)
        
        with col1:
            next_date = latest_date + timedelta(days=1)
            # Keep incrementing by 1 day if it falls on a weekend
            while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                next_date += timedelta(days=1)
                
            selected_date = st.date_input(
                "Next Working Day",
                value=next_date,
                min_value=latest_date + timedelta(days=1),
                help="Select the next working day for demand adjustment"
            )
            
            # Display day of week for clarity
            st.write(f"Selected: {selected_date.strftime('%A, %B %d, %Y')}")
            
        # Work Type selector
        available_work_types = sorted(df['WorkType'].unique())
        
        selected_work_type = st.selectbox(
            "Select Work Type",
            options=available_work_types,
            index=0 if available_work_types else None,
            help="Select the work type to update demand for"
        )
        
        # Demand parameters
        st.subheader("Demand Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            backlog_percentage = st.slider(
                "Backlog (%)",
                min_value=0,
                max_value=100,
                value=25,
                step=5,
                help="Percentage of previous day's work that remains as backlog"
            )
        
        with col2:
            new_demand = st.number_input(
                "New Demand",
                min_value=0,
                value=0,
                step=1,
                help="New demand for the selected day (number of workers)"
            )
        
        # Summary box
        st.subheader("Demand Summary")
        
        # Calculate expected workers based on backlog and new demand
        # Get the previous day's data for the selected work type
        prev_day = selected_date - timedelta(days=1)
        prev_day_data = df[(df['Date'] == pd.Timestamp(prev_day)) & (df['WorkType'] == selected_work_type)]
        
        prev_day_workers = prev_day_data['NoOfMan'].sum() if not prev_day_data.empty else 0
        expected_backlog = (prev_day_workers * backlog_percentage / 100)
        total_expected_workers = expected_backlog + new_demand
        
        # Display summary in a nice box
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.metric("Previous Day Workers", f"{prev_day_workers:.0f}")
            st.metric("Expected Backlog", f"{expected_backlog:.2f}")
            
        with summary_col2:
            st.metric("New Demand", f"{new_demand:.0f}")
            st.metric("Total Expected Workers", f"{total_expected_workers:.2f}")
        
        # Save button
        if st.button("Save Demand Parameters", type="primary"):
            # Logic to save demand parameters would go here
            # This would typically update a configuration file or database
            
            st.success(f"Demand parameters saved for {selected_work_type} on {selected_date.strftime('%Y-%m-%d')}")
            
            # Add code here to store the updated demand parameters
            # You might want to create a new dataframe or update an existing one
            
            # For demonstration, we'll just show what would be saved
            st.write("Saved parameters:")
            
            saved_data = pd.DataFrame({
                'Date': [selected_date],
                'WorkType': [selected_work_type],
                'Backlog (%)': [backlog_percentage],
                'New Demand': [new_demand],
                'Total Expected': [total_expected_workers]
            })
            
            st.dataframe(saved_data)
            
            # Download option
            st.download_button(
                label="Download Demand Parameters (CSV)",
                data=saved_data.to_csv(index=False),
                file_name=f"demand_parameters_{selected_date.strftime('%Y%m%d')}_{selected_work_type}.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.subheader("Demand History")
        
        # This would typically display historical demand adjustments
        # For now, we'll just show a placeholder
        
        st.info("Historical demand adjustments will be displayed here.")
        
        # Mock data for visualization
        dates = pd.date_range(end=latest_date, periods=10)
        mock_history = pd.DataFrame({
            'Date': dates,
            'Backlog (%)': np.random.randint(10, 40, size=10),
            'New Demand': np.random.randint(5, 20, size=10),
            'Total': np.random.randint(15, 50, size=10)
        })
        
        # Display mock data
        st.dataframe(mock_history)
        
        # Visualization
        st.subheader("Demand Trend")
        
        # Create figure
        fig = go.Figure()
        
        # Add backlog line
        fig.add_trace(go.Scatter(
            x=mock_history['Date'],
            y=mock_history['Backlog (%)'],
            mode='lines+markers',
            name='Backlog (%)',
            line=dict(color='orange', width=2)
        ))
        
        # Add new demand line
        fig.add_trace(go.Scatter(
            x=mock_history['Date'],
            y=mock_history['New Demand'],
            mode='lines+markers',
            name='New Demand',
            line=dict(color='green', width=2)
        ))
        
        # Add total line
        fig.add_trace(go.Scatter(
            x=mock_history['Date'],
            y=mock_history['Total'],
            mode='lines+markers',
            name='Total Expected',
            line=dict(color='blue', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Demand Trend Over Time',
            xaxis_title='Date',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)