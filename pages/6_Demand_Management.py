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
import pyodbc

from utils.demand_scheduler import DemandScheduler, shift_demand_forward, get_next_working_day

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_manager import StateManager
from utils.kpi_manager import load_punch_codes
from utils.sql_data_connector import load_demand_forecast_data, SQLDataConnector
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

def load_actual_quantities_for_tomorrow():
    """
    Load actual quantities for tomorrow using the simplified query
    """
    try:
        # Connect to SQL Server
        conn = SQLDataConnector.connect_to_sql(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if not conn:
            logger.error("Failed to connect to database")
            return {}
        
        cursor = conn.cursor()
        
        # Get tomorrow's date
        tomorrow = get_next_working_day(datetime.today())
        print(f'Nexr Working is {tomorrow}')
        # Execute the simplified query with hardcoded date to avoid parameter issues
        query = f"""
        DECLARE @Tomorrow DATE = '{tomorrow.strftime('%Y-%m-%d')}';
        
        SELECT 
            @Tomorrow AS PlanDate,
            SUM(reqquant - delquant) AS Quantity,
            pc.Punchcode
        FROM fsystemp.dbo.O08T1
        JOIN fsystemp.dbo.R08T1 
            ON O08T1.shortr08 = R08T1.shortr08
        OUTER APPLY (
            SELECT CASE
                WHEN routeno = 'MÄSSA' THEN '207'
                WHEN routeno LIKE 'N[12]Z%' OR routeno LIKE '[12]Z%' THEN '209'
                WHEN routeno IN ('SORT1', 'SORTP1') THEN '209'
                WHEN routeno IN (
                    'BOOZT', 'ÅHLENS', 'AMZN', 'ENS1', 'ENS2', 'EMV', 'EXPRES', 'KLUBB', 
                    'ÖP', 'ÖPFAPO', 'ÖPLOCK', 'ÖPSPEC', 'ÖPUTRI', 'PRINTW', 'RLEV'
                ) THEN '211'
                WHEN routeno IN ('LÄROME', 'SORDER', 'FSMAK', 'ORKLA', 'REAAKB', 'REAUGG') THEN '214'
                WHEN routeno IN ('ADLIB', 'BIB', 'BOKUS', 'DIVNÄT', 'BUYERS') THEN '215'
                WHEN divcode IN ('LIB', 'NYP', 'STU') THEN '213'
                ELSE '211'
            END AS Punchcode
        ) pc
        WHERE linestat IN (2, 4, 22, 30)
          AND R08T1.oppdate <= @Tomorrow
        GROUP BY pc.Punchcode
        ORDER BY pc.Punchcode;
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Convert to dictionary
        actual_quantities = {}
        for row in results:
            punch_code = str(row[2])  # Punchcode
            quantity = int(round(float(row[1]))) if row[1] is not None else 0  # Round to integer
            actual_quantities[punch_code] = quantity
        
        cursor.close()
        conn.close()
        
        logger.info(f"Loaded actual quantities for tomorrow ({tomorrow}): {actual_quantities}")
        return actual_quantities
        
    except Exception as e:
        logger.error(f"Error loading actual quantities: {str(e)}")
        return {}

def format_cell_value(value, cell_id, actual_cells, modified_cells):
    """
    Format cell value with appropriate symbol and styling
    """
    if cell_id in actual_cells:
        return f"✓ {value}"  # Actual quantities with checkmark and bold
    elif cell_id in modified_cells:
        return f"● {value}"  # User modified with bullet and bold
    else:
        return str(value)  # Original data (plain number)

def extract_numeric_value(formatted_value):
    """
    Extract numeric value from formatted string
    """
    if isinstance(formatted_value, str):
        # Remove symbols and markdown formatting
        clean_value = formatted_value.replace('✓', '').replace('●', '').replace('**', '').strip()
        try:
            return int(clean_value)
        except ValueError:
            return 0
    return int(formatted_value)

# def create_formatted_dataframe(df, actual_cells, modified_cells, punch_code_values):
#     """
#     Create a dataframe with formatted cell values
#     """
#     formatted_df = df.copy()
    
#     for idx, row in formatted_df.iterrows():
#         date_str = row['Date']
#         for punch_code in punch_code_values:
#             cell_id = f"{date_str}_{punch_code}"
#             original_value = row[punch_code]
#             formatted_value = format_cell_value(original_value, cell_id, actual_cells, modified_cells)
#             formatted_df.at[idx, punch_code] = formatted_value
    
#     return formatted_df

def create_formatted_dataframe(df, actual_cells, modified_cells, punch_code_values):
    """
    Create a dataframe with formatted cell values and proper string column types
    """
    formatted_df = df.copy()
    
    # Convert punch code columns to string type to avoid dtype warnings
    for punch_code in punch_code_values:
        formatted_df[punch_code] = formatted_df[punch_code].astype(str)
    
    for idx, row in formatted_df.iterrows():
        date_str = row['Date']
        for punch_code in punch_code_values:
            cell_id = f"{date_str}_{punch_code}"
            original_value = row[punch_code]
            formatted_value = format_cell_value(original_value, cell_id, actual_cells, modified_cells)
            formatted_df.at[idx, punch_code] = str(formatted_value)  # Explicitly convert to string
    
    return formatted_df

def save_forecast_to_database(demand_df, punch_code_values, username):
    """
    Save forecast data to PredictionData table using stored procedure for all calculations and data saving
    """
    try:
        # Connect to SQL Server
        conn = SQLDataConnector.connect_to_sql(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if not conn:
            logger.error("Failed to connect to database")
            return False, "Failed to connect to database"
        
        cursor = conn.cursor()
        
        # Counter for successful saves
        save_count = 0
        error_count = 0
        
        # Process each row (date)
        for _, row in demand_df.iterrows():
            date_str = row['Date']
            
            # Extract date from the format "YYYY-MM-DD (Day)"
            try:
                actual_date = date_str.split(' (')[0]  # Remove day name part
                date_obj = datetime.strptime(actual_date, DATE_FORMAT).date()
            except ValueError as e:
                logger.error(f"Error parsing date '{date_str}': {e}")
                error_count += 1
                continue
            
            # Process each punch code
            for punch_code in punch_code_values:
                quantity = extract_numeric_value(row[punch_code])  # Extract numeric value
                
                # Only save if quantity is greater than 0
                if quantity > 0:
                    try:
                        punch_code_int = int(punch_code)
                        
                        # Use explicit parameter types to avoid binding issues
                        cursor.execute("""
                            DECLARE @Result INT;
                            EXEC @Result = usp_SaveForecastWithKPICalculation ?, ?, ?, ?;
                            """, 
                            date_obj.strftime('%Y-%m-%d'),  # Convert date to string
                            punch_code_int, 
                            float(quantity), 
                            str(username)
                        )
                        
                        save_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error saving prediction for date={date_obj}, punch_code={punch_code}: {str(e)}")
                        error_count += 1
                        continue
        
        # Commit all changes
        conn.commit()
        
        # Close connection
        cursor.close()
        conn.close()
        
        if save_count > 0:
            return True, f"Successfully saved {save_count} forecast entries to database"
        else:
            return False, f"No entries were saved. Errors: {error_count}"
        
    except Exception as e:
        logger.error(f"Error saving forecast to database: {str(e)}")
        return False, f"Database error: {str(e)}"

def get_current_user():
    """Get the current user from the system"""
    try:
        import os
        import getpass
        username = os.environ.get('USERNAME', getpass.getuser())
        return username
    except:
        return "unknown"

def load_forecast_data_automatically(today, end_date, dates, day_names, punch_code_values):
    """
    Load forecast data automatically when page loads
    """
    try:
        forecast_df = load_demand_forecast_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )

        if forecast_df is None or forecast_df.empty:
            # Return empty dataframe if no data
            rows = []
            for i, date in enumerate(dates):
                row = {"Date": f"{date} ({day_names[i]})"}
                for code in punch_code_values:
                    row[code] = 0
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            # Convert PlanDate to datetime
            forecast_df['PlanDate'] = pd.to_datetime(forecast_df['PlanDate'])
            
            # Convert Punchcode to string to match punch_code_values
            forecast_df['Punchcode'] = forecast_df['Punchcode'].astype(str)
            
            # Filter forecast data to match our 7-day period
            start_date = pd.Timestamp(today)
            end_date_ts = pd.Timestamp(end_date)
            
            filtered_forecast = forecast_df[
                (forecast_df['PlanDate'] >= start_date) & 
                (forecast_df['PlanDate'] <= end_date_ts)
            ]
            print('demand ', filtered_forecast.head(100))
            if not filtered_forecast.empty:
                # Create pivot table with PlanDate as index and Punchcode as columns
                pivot_df = filtered_forecast.pivot_table(
                    index='PlanDate',
                    columns='Punchcode',
                    values='Quantity',
                    fill_value=0.0
                ).reset_index()
                
                # Create the demand dataframe structure
                rows = []
                for i, date in enumerate(dates):
                    row = {"Date": f"{date} ({day_names[i]})"}
                    
                    # Match the date with pivot data
                    date_obj = datetime.strptime(date, DATE_FORMAT).date()
                    matching_pivot_row = pivot_df[pivot_df['PlanDate'].dt.date == date_obj]
                    
                    # Populate punch code values (rounded to integers)
                    for code in punch_code_values:
                        if not matching_pivot_row.empty and code in pivot_df.columns:
                            row[code] = int(round(float(matching_pivot_row[code].iloc[0])))
                        else:
                            row[code] = 0
                    
                    rows.append(row)
                
                return pd.DataFrame(rows)
            else:
                # Return empty dataframe if no data for date range
                rows = []
                for i, date in enumerate(dates):
                    row = {"Date": f"{date} ({day_names[i]})"}
                    for code in punch_code_values:
                        row[code] = 0
                    rows.append(row)
                return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Error loading forecast data: {str(e)}")
        # Return empty dataframe on error
        rows = []
        for i, date in enumerate(dates):
            row = {"Date": f"{date} ({day_names[i]})"}
            for code in punch_code_values:
                row[code] = 0
            rows.append(row)
        return pd.DataFrame(rows)

def calculate_totals(df, actual_cells, modified_cells, punch_code_values):
    """
    Calculate totals for different data types
    """
    original_total = 0
    actual_total = 0
    modified_total = 0
    
    for idx, row in df.iterrows():
        date_str = row['Date']
        for punch_code in punch_code_values:
            cell_id = f"{date_str}_{punch_code}"
            value = extract_numeric_value(row[punch_code])
            
            if cell_id in actual_cells:
                actual_total += value
            elif cell_id in modified_cells:
                modified_total += value
            else:
                original_total += value
    
    return original_total, actual_total, modified_total

def main():
    st.header("📊 Demand Management")
    
   
    # Get today's date
    today = datetime.now().date()
    
    # Calculate 7 days from today
    end_date = today + timedelta(days=13)  # 7 days including today
    
    # Display date range using date format from config
    st.subheader(f"Demand Forecast: {today.strftime(DATE_FORMAT)} to {end_date.strftime(DATE_FORMAT)}")
    
    # Load punch codes
    punch_codes = load_punch_codes()
    punch_code_values = [str(pc["value"]) for pc in punch_codes]
    
    # Create empty dataframe for the next 7 days
    date_range = pd.date_range(start=today, periods=14)
    dates = [d.strftime(DATE_FORMAT) for d in date_range]
    
    # Add day names for better visibility
    day_names = [d.strftime("%a") for d in date_range]
    date_labels = [f"{date} ({day})" for date, day in zip(dates, day_names)]
    
    # Initialize session state for demand data and load data automatically
    if 'demand_df' not in st.session_state or not st.session_state.get('demand_data_loaded', False):
        with st.spinner("Loading forecast data..."):
            st.session_state.demand_df = load_forecast_data_automatically(
                today, end_date, dates, day_names, punch_code_values
            )
            st.session_state.demand_data_loaded = True
            
            # Initialize tracking sets
            if 'actual_quantities_cells' not in st.session_state:
                st.session_state.actual_quantities_cells = set()
            if 'user_modified_cells' not in st.session_state:
                st.session_state.user_modified_cells = set()

    st.write("### Summary")
    col1, col2 = st.columns(2)
             
    with col1:
        # Load actual quantities button
        if st.button("🟢 Load  Next Working Day Quantities", type="secondary"):
            with st.spinner("Loading actual quantities for tomorrow..."):
                actual_quantities = load_actual_quantities_for_tomorrow()
                print(actual_quantities)
                if actual_quantities:
                    # Update tomorrow's row with actual quantities
                    tomorrow = get_next_working_day(today)
                    tomorrow_str = f"{tomorrow.strftime(DATE_FORMAT)} ({tomorrow.strftime('%a')})"

                    # Find tomorrow's row
                    tomorrow_row_idx = st.session_state.demand_df[
                        st.session_state.demand_df['Date'] == tomorrow_str
                    ].index
                    
                    if not tomorrow_row_idx.empty:
                        idx = tomorrow_row_idx[0]
                        
                        # Update quantities and store which cells were updated with actual data
                        for punch_code, quantity in actual_quantities.items():
                            if punch_code in punch_code_values:
                                st.session_state.demand_df.at[idx, punch_code] = quantity
                                # Mark this cell as having actual data (for checkmark formatting)
                                st.session_state.actual_quantities_cells.add(f"{tomorrow_str}_{punch_code}")
                                # Remove from modified cells if it was there
                                st.session_state.user_modified_cells.discard(f"{tomorrow_str}_{punch_code}")
                        
                        st.success(f"✅ Loaded actual quantities for tomorrow ({tomorrow.strftime('%Y-%m-%d')}) - {len(actual_quantities)} punch codes updated")
                        st.rerun()
                    else:
                        st.error("Could not find tomorrow's row in the forecast data")
                else:
                    st.warning("No actual quantities found for tomorrow")

    with col2:
        # Add demand shift button
        if st.button("🔄 Auto-Shift Non-Working Day Demand", type="secondary"):
            with st.spinner("Analyzing and shifting demand from non-working days..."):
                scheduler = DemandScheduler()
                
                # Shift demand forward to next working days, including visual indicators
                updated_df, shift_log, updated_actual_cells, updated_modified_cells = scheduler.shift_demand_for_non_working_day(
                    st.session_state.demand_df, 
                    punch_code_values, 
                    "forward",
                    st.session_state.actual_quantities_cells,
                    st.session_state.user_modified_cells
                )
                
                if shift_log:
                    # Update the dataframe and tracking sets
                    st.session_state.demand_df = updated_df
                    st.session_state.actual_quantities_cells = updated_actual_cells
                    st.session_state.user_modified_cells = updated_modified_cells
                    
                    # Generate and display summary report
                    summary_report = scheduler.generate_shift_summary_report(shift_log)
                    
                    st.success(f"✅ Demand shifted successfully! Processed {len(shift_log)} non-working days.")
                    
                    # Show detailed report in expandable section
                    with st.expander("📊 View Detailed Shift Report", expanded=True):
                        st.text(summary_report)
                    
                    # Force refresh to show updated data
                    st.rerun()
                else:
                    st.info("ℹ️ No demand shifts were needed - all dates are working days.")
                
    # Tabs for different views
    tab1, tab2 = st.tabs(["Demand Forecast", "Adjustment Factors"])
    
    with tab1:
        # Create formatted dataframe for display
        display_df = create_formatted_dataframe(
            st.session_state.demand_df,
            st.session_state.actual_quantities_cells,
            st.session_state.user_modified_cells,
            punch_code_values
        )
        
        # Make dataframe editable with formatted values
        st.write("### Editable Forecast with Visual Indicators")
        edited_df = st.data_editor(
            display_df,
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
                    code: st.column_config.TextColumn(  # Use TextColumn to preserve formatting
                        code,
                        width="small",
                        help=f"Forecast for Punch Code {code}"
                    )
                    for code in punch_code_values
                }
            },
            key="demand_editor"
        )

        # Update session state with edited values and track changes
        if not display_df.equals(edited_df):
            changes_made = False  # Track if any changes were made
            
            # Process changes and update the underlying data
            for idx, row in edited_df.iterrows():
                for col in punch_code_values:
                    if display_df.at[idx, col] != row[col]:
                        cell_id = f"{row['Date']}_{col}"
                        
                        # Extract numeric value and update the base dataframe
                        new_value = extract_numeric_value(row[col])
                        st.session_state.demand_df.at[idx, col] = new_value
                        
                        # Mark as user-modified if it's not an actual quantity cell
                        if cell_id not in st.session_state.actual_quantities_cells:
                            st.session_state.user_modified_cells.add(cell_id)
                            changes_made = True
            
            # Force a rerun to immediately update the display with new formatting
            if changes_made:
                st.rerun()

        # MOVE THE METRICS CALCULATION HERE - AFTER DATA PROCESSING
        # Calculate totals for color-coded metrics
        original_total, actual_total, modified_total = calculate_totals(
            st.session_state.demand_df,  # Use the underlying dataframe, not the formatted one
            st.session_state.actual_quantities_cells,
            st.session_state.user_modified_cells,
            punch_code_values
        )
        
        # Color-coded metric boxes
        st.write("### Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔵 Original Forecast", f"{original_total:,}", help="Total from database forecast")
        
        with col2:
            st.metric("🟢 Actual Quantities", f"{actual_total:,}", help="Total from tomorrow's actual data")
        
        with col3:
            st.metric("🔴 Your Changes", f"{modified_total:,}", help="Total from your modifications")
        
        with col4:
            total_all = original_total + actual_total + modified_total
            st.metric("📊 Grand Total", f"{total_all:,}", help="Sum of all quantities")
        
        # Legend
        st.markdown("""
        **Legend:**
        - **Plain numbers**: Original forecast data loaded from database
        - **✓ Bold numbers**: Tomorrow's actual quantities
        - **● Bold numbers**: User-modified values
        """)
        
        # Display information about data types
        if st.session_state.actual_quantities_cells:
            st.info(f"✅ {len(st.session_state.actual_quantities_cells)} cells contain actual quantities for tomorrow")
        
        if st.session_state.user_modified_cells:
            st.warning(f"✏️ {len(st.session_state.user_modified_cells)} cells have been modified by you")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            # Username input
            username = st.text_input("Your Username", value=get_current_user())
        
        with col2:
            # Save Forecast button
            if st.button("Save Forecast", type="primary"):
                if not username:
                    st.error("Please enter your username")
                else:
                    with st.spinner("Saving forecast data..."):
                        # Use the original dataframe for saving (with numeric values)
                        success, message = save_forecast_to_database(
                            st.session_state.demand_df,  # Use the underlying dataframe instead of display_df
                            punch_code_values, 
                            username
                        )
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
            
        with col3:
            st.write("")

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
    
if __name__ == "__main__":
    main()
    StateManager.initialize()