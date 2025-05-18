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
import io

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_manager import StateManager
from utils.kpi_manager import load_punch_codes
from utils.sql_data_connector import load_demand_forecast_data
from config import (DATA_DIR, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, 
                   DATE_FORMAT, CACHE_TTL, SQL_DATABASE_LIVE)

# Configure page
st.set_page_config(
    page_title="Demand Management",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def handle_data_edit(edited_df):
    """
    Callback to handle data edits and update session state
    """
    st.session_state.kpi_df = edited_df
    st.session_state.has_unsaved_changes = True

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
    
    # Display date range using date format from config
    st.subheader(f"Demand Forecast: {today.strftime(DATE_FORMAT)} to {end_date.strftime(DATE_FORMAT)}")
    
    # Load punch codes
    punch_codes = load_punch_codes()
    punch_code_values = [str(pc["value"]) for pc in punch_codes]
    
    # Create empty dataframe for the next 7 days
    date_range = pd.date_range(start=today, periods=7)
    dates = [d.strftime(DATE_FORMAT) for d in date_range]
    
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
    
    # Initialize session state for demand data if not already present
    if 'demand_df' not in st.session_state:
        # Create dataframe with days as rows and punch codes as columns
        rows = []
        for i, date in enumerate(dates):
            row = {"Date": f"{date} ({day_names[i]})"}  # Include day name in the date
            for code in punch_code_values:
                row[code] = 0.0  # Initialize with zeros
            rows.append(row)
        
        st.session_state.demand_df = pd.DataFrame(rows)
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["Demand Forecast", "Adjustment Factors"])
    
    with tab1:
        # Make dataframe editable
        edited_df = st.data_editor(
            st.session_state.demand_df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Date": st.column_config.TextColumn(
                    "Date",
                    width="medium",
                    help="Forecast date"
                ),
                **{
                    code: st.column_config.NumberColumn(
                        code,
                        format="%.1f",
                        width="small",
                        help=f"Forecast for Punch Code {code}"
                    )
                    for code in punch_code_values
                }
            },
            key="demand_editor"
        )

        
        # Update session state with edited values
        st.session_state.demand_df = edited_df
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            # Load Forecast button
            if st.button("Load Forecast", type="primary"):
                with st.spinner("Loading forecast data from database..."):
                    try:
                        # First add a database selection
                        if 'target_database' not in st.session_state:
                            st.session_state.target_database = SQL_DATABASE_LIVE
                        
                        # Show the database selection
                        database_options = {
                            SQL_DATABASE: f"Primary Database ({SQL_DATABASE})",
                            "fsystemp": "FSysTemp Database"
                        }
                        
                        selected_database = st.radio(
                            "Select Database",
                            options=list(database_options.keys()),
                            format_func=lambda x: database_options[x],
                            index=list(database_options.keys()).index(st.session_state.target_database),
                            key="database_selector"
                        )
                        
                        st.session_state.target_database = selected_database
                        
                        # Load forecast data from SQL
                        with st.status("Connecting to database...") as status:
                            forecast_df = load_demand_forecast_data(
                                server=SQL_SERVER, 
                                database=st.session_state.target_database,
                                trusted_connection=SQL_TRUSTED_CONNECTION
                            )
                            
                            if forecast_df is not None and not forecast_df.empty:
                                status.update(label="Processing forecast data...", state="running")
                                
                                # Convert forecast data to the format we need (date rows, punch code columns)
                                # First, ensure PlanDate is datetime
                                forecast_df['PlanDate'] = pd.to_datetime(forecast_df['PlanDate'])
                                
                                # Filter to only include dates in our range
                                min_date = pd.to_datetime(today)
                                max_date = pd.to_datetime(end_date)
                                filtered_forecast = forecast_df[
                                    (forecast_df['PlanDate'] >= min_date) & 
                                    (forecast_df['PlanDate'] <= max_date)
                                ]
                                
                                if not filtered_forecast.empty:
                                    # Create a new dataframe with our structure
                                    new_rows = []
                                    for i, date_obj in enumerate(date_range):
                                        date_str = dates[i]
                                        day_name = day_names[i]
                                        
                                        row = {"Date": f"{date_str} ({day_name})"}
                                        
                                        # Add each punch code
                                        for code in punch_code_values:
                                            # Find matching forecast data
                                            matching_data = filtered_forecast[
                                                (filtered_forecast['PlanDate'] == date_obj) & 
                                                (filtered_forecast['Punchcode'] == code)
                                            ]
                                            
                                            if not matching_data.empty:
                                                # Use the quantity from forecast
                                                row[code] = float(matching_data['Quantity'].iloc[0])
                                            else:
                                                # No forecast for this punch code on this date
                                                row[code] = 0.0
                                        
                                        new_rows.append(row)
                                    
                                    # Update session state with new data
                                    st.session_state.demand_df = pd.DataFrame(new_rows)
                                    status.update(label="Forecast data loaded successfully!", state="complete")
                                    st.rerun()  # Rerun to update the UI
                                else:
                                    status.update(label="No forecast data found for selected dates", state="error")
                                    st.warning("No forecast data found for the selected date range.")
                            else:
                                status.update(label="Failed to load forecast data", state="error")
                                st.error(f"Failed to load forecast data from {st.session_state.target_database} database.")
                                st.info("Please check the server logs for more details or try a different database.")
                    except Exception as e:
                        st.error(f"Error loading forecast data: {str(e)}")
                        logger.error(f"Error loading forecast data: {str(e)}")
                        logger.error(traceback.format_exc())

        
        with col2:
            # Save Forecast button
            if st.button("Save Forecast"):
                st.info("Forecast saving functionality will be implemented in a future update.")
                
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
    
    # Calculate totals for each day (sum across punch codes)
    totals = {}
    for i, date in enumerate(dates):
        # Sum all punch code values for this date (row)
        day_total = sum([
            float(st.session_state.demand_df.loc[i, code]) 
            for code in punch_code_values 
            if code in st.session_state.demand_df.columns
        ])
        totals[date] = day_total
    
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
    
    # Export to Excel
    if st.button("Export to Excel", type="secondary"):
        try:
            # Create Excel buffer
            buffer = io.BytesIO()
            
            # Create Excel writer
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Write demand forecast
                st.session_state.demand_df.to_excel(writer, sheet_name='Demand Forecast', index=False)
                
                # Write capacity analysis
                capacity_df.to_excel(writer, sheet_name='Capacity Analysis', index=False)
            
            buffer.seek(0)
            
            # Offer download
            st.download_button(
                label="Download Excel File",
                data=buffer,
                file_name=f"demand_forecast_{today.strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download the demand forecast data as an Excel file"
            )
        except Exception as e:
            st.error(f"Error exporting to Excel: {str(e)}")
            logger.error(f"Error exporting to Excel: {str(e)}")
            logger.error(traceback.format_exc())

    
if __name__ == "__main__":
    main()
    StateManager.initialize()