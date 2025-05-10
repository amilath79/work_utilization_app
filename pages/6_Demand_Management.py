"""
Demand Management page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_manager import StateManager
from utils.kpi_manager import load_punch_codes
from config import DATA_DIR, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION

# Configure page
st.set_page_config(
    page_title="Demand Management",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

def main():
    st.header("📊 Demand Management")
    
    st.info("""
    This page allows you to manage and adjust demand forecasts for the upcoming 7 days.
    Review the predicted workforce requirements and make adjustments as needed.
    """)
    
    # Get today's date
    today = datetime.now().date()
    
    # Calculate 7 days from today
    end_date = today + timedelta(days=6)  # 7 days including today
    
    # Display date range
    st.subheader(f"Demand Forecast: {today.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Load punch codes
    punch_codes = load_punch_codes()
    punch_code_values = [str(pc["value"]) for pc in punch_codes]
    
    # Create empty dataframe for the next 7 days
    date_range = pd.date_range(start=today, periods=7)
    dates = [d.strftime("%Y-%m-%d") for d in date_range]
    
    # Add day names for better visibility
    day_names = [d.strftime("%a") for d in date_range]
    date_labels = [f"{date} ({day})" for date, day in zip(dates, day_names)]
    
    # Create columns for metrics
    st.write("### Summary")
    cols = st.columns(7)
    for i, col in enumerate(cols):
        with col:
            st.metric(
                label=day_names[i],
                value=dates[i],
                delta="0"  # Placeholder for change in demand
            )
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["Demand Forecast", "Adjustment Factors"])
    
    with tab1:
        # Create dataframe with days as columns and punch codes as rows
        rows = []
        for code in punch_code_values:
            row = {"Punch Code": code}
            for date in dates:
                row[date] = 0.0  # Initialize with zeros
            rows.append(row)
        
        demand_df = pd.DataFrame(rows)
        
        # Make dataframe editable
        edited_df = st.data_editor(
            demand_df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Punch Code": st.column_config.TextColumn(
                    "Punch Code",
                    width="medium",
                    help="Punch code identifier"
                ),
                **{
                    date: st.column_config.NumberColumn(
                        day_name,
                        format="%.1f",
                        width="small",
                        help=f"Forecast for {date}"
                    )
                    for date, day_name in zip(dates, day_names)
                }
            }
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            st.button("Load Forecast", type="primary")
        with col2:
            st.button("Save Forecast")
        with col3:
            st.write("")  # Empty column for spacing
    
    with tab2:
        st.write("### Adjustment Factors")
        
        # Create columns for different factors
        adj_col1, adj_col2 = st.columns(2)
        
        with adj_col1:
            st.write("#### Seasonal Factors")
            
            # Seasonal adjustment sliders
            st.slider("Weekend Adjustment", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
            st.slider("Holiday Adjustment", min_value=0.5, max_value=1.5, value=0.8, step=0.1)
            st.slider("Monday Adjustment", min_value=0.5, max_value=1.5, value=1.1, step=0.1)
            st.slider("Friday Adjustment", min_value=0.5, max_value=1.5, value=0.9, step=0.1)
        
        with adj_col2:
            st.write("#### Operational Factors")
            
            # Operational adjustment sliders
            st.slider("Backlog Factor", min_value=0.0, max_value=1.0, value=0.2, step=0.05, 
                     help="Percentage of previous day's work that remains as backlog")
            st.slider("Productivity Factor", min_value=0.8, max_value=1.2, value=1.0, step=0.05)
            st.slider("Absence Factor", min_value=0.0, max_value=0.2, value=0.05, step=0.01, 
                     help="Expected absence rate")
    
    # Summary section
    st.write("### Total Workforce Requirements")
    
    # Calculate totals (this would be dynamic in the real implementation)
    totals = {date: np.sum([float(edited_df.at[i, date]) for i in range(len(edited_df))]) for date in dates}
    
    # Create a bar chart of total requirements
    totals_df = pd.DataFrame({
        "Date": day_names,
        "Workers Required": list(totals.values())
    })
    
    st.bar_chart(totals_df.set_index("Date"))
    
    # Comparison to capacity
    st.write("### Capacity Analysis")
    
    capacity_df = pd.DataFrame({
        "Date": day_names,
        "Forecast": list(totals.values()),
        "Capacity": [100, 100, 100, 100, 80, 50, 80]  # Example capacity values
    })
    
    capacity_df["Utilization"] = (capacity_df["Forecast"] / capacity_df["Capacity"] * 100).round(1)
    capacity_df["Excess/Shortfall"] = (capacity_df["Capacity"] - capacity_df["Forecast"]).round(1)
    
    st.dataframe(
        capacity_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Day"),
            "Forecast": st.column_config.NumberColumn("Forecast", format="%.1f"),
            "Capacity": st.column_config.NumberColumn("Capacity", format="%.1f"),
            "Utilization": st.column_config.ProgressColumn(
                "Utilization",
                format="%.1f%%",
                min_value=0,
                max_value=150
            ),
            "Excess/Shortfall": st.column_config.NumberColumn(
                "Excess/Shortfall", 
                format="%.1f",
                help="Positive values indicate excess capacity, negative values indicate shortfall"
            )
        }
    )
    
    # Action button
    st.button("Export to Excel", type="secondary")

if __name__ == "__main__":
    main()
    StateManager.initialize()