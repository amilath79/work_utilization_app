"""
Next Day Prediction Accuracy page - Simplified version with Workers and Hours
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import sys
import traceback

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_manager import StateManager
from utils.sql_data_connector import extract_sql_data, load_demand_with_kpi_data
from utils.demand_scheduler import get_next_working_day
from config import SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, SQL_DATABASE_LIVE

# Configure page
st.set_page_config(
    page_title="Next Day Prediction Accuracy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

def load_prediction_data(date_value):
    """Load prediction data from the PredictionData table"""
    try:
        sql_query = f"""
        SELECT ID, Date, PunchCode, NoOfMan, Hours, PredictionType, Username, 
               CreatedDate, LastModifiedDate
        FROM PredictionData WHERE PunchCode in (209,211, 213, 214, 215, 202, 203, 206, 210, 217)
        AND Date = '{date_value}'
        ORDER BY PunchCode
        """
        
        df = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=sql_query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        
        if df is not None and not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df['PunchCode'] = df['PunchCode'].astype(str)
            return df
        return None
    except Exception as e:
        logger.error(f"Error loading prediction data: {str(e)}")
        return None

def calculate_improved_prediction(prediction_df, target_date):
    """Calculate improved predictions for both workers and hours"""
    try:
        improved_workers = {}
        improved_hours = {}
        
        # Get demand data with KPI
        next_working_day = get_next_working_day(datetime.now().date())
        demand_kpi_df = load_demand_with_kpi_data(next_working_day.strftime('%Y-%m-%d'))
        
        if demand_kpi_df is not None and not demand_kpi_df.empty:
            target_demand_data = demand_kpi_df[demand_kpi_df['PlanDate'].dt.date == target_date]
            
            # Demand-based punch codes
            demand_codes = ['209', '211', '213', '215']
            for punch_code in demand_codes:
                punch_data = target_demand_data[target_demand_data['Punchcode'] == punch_code]
                
                if not punch_data.empty:
                    if punch_code in ['206', '213']:
                        quantity = punch_data['nrows'].sum()
                    else:
                        quantity = punch_data['Quantity'].sum()
                    
                    kpi_value = punch_data['KPIValue'].iloc[0]
                    
                    if quantity > 0 and kpi_value > 0:
                        workers = max(quantity / kpi_value / 8, 0)
                        hours = workers * 8
                    else:
                        workers = 0
                        hours = 0
                    
                    improved_workers[punch_code] = round(workers, 1)
                    improved_hours[punch_code] = round(hours, 1)
                else:
                    improved_workers[punch_code] = 0
                    improved_hours[punch_code] = 0
        
        # ML-based punch codes (use 95% of original prediction)
        ml_codes = ['202', '203', '206', '210', '214',  '217']
        for punch_code in ml_codes:
            if prediction_df is not None and not prediction_df.empty:
                punch_predictions = prediction_df[prediction_df['PunchCode'] == punch_code]
                
                if not punch_predictions.empty:
                    original_workers = punch_predictions['NoOfMan'].iloc[0]
                    workers = max(original_workers * 0.95, 0)
                    hours = workers * 8
                    
                    improved_workers[punch_code] = round(workers, 1)
                    improved_hours[punch_code] = round(hours, 1)
                else:
                    improved_workers[punch_code] = 0
                    improved_hours[punch_code] = 0
        
        return improved_workers, improved_hours
        
    except Exception as e:
        logger.error(f"Error calculating improved prediction: {str(e)}")
        return {}, {}

def create_comparison_dataframe(prediction_df, improved_workers, improved_hours, target_date):
    """Create comparison dataframe with both metrics"""
    try:
        if isinstance(target_date, datetime):
            target_date_dt = target_date.date()
        else:
            target_date_dt = target_date
            
        target_predictions = prediction_df[prediction_df['Date'].dt.date == target_date_dt]
        
        comparison_data = []
        
        # Process existing predictions
        for _, row in target_predictions.iterrows():
            punch_code = row['PunchCode']
            original_workers = row['NoOfMan']
            original_hours = row['Hours'] if 'Hours' in row else original_workers * 8
            
            new_workers = improved_workers.get(punch_code, 0)
            new_hours = improved_hours.get(punch_code, 0)
            
            comparison_data.append({
                'PunchCode': punch_code,
                'Original Workers': original_workers,
                'Improved Workers': new_workers,
                'Original Hours': original_hours,
                'Improved Hours': new_hours,
                'Workers Change': new_workers - original_workers,
                'Hours Change': new_hours - original_hours
            })
        
        # Add any punch codes only in improved predictions
        for punch_code in improved_workers.keys():
            if punch_code not in target_predictions['PunchCode'].values:
                comparison_data.append({
                    'PunchCode': punch_code,
                    'Original Workers': 0,
                    'Improved Workers': improved_workers[punch_code],
                    'Original Hours': 0,
                    'Improved Hours': improved_hours[punch_code],
                    'Workers Change': improved_workers[punch_code],
                    'Hours Change': improved_hours[punch_code]
                })
        
        return pd.DataFrame(comparison_data)
        
    except Exception as e:
        logger.error(f"Error creating comparison dataframe: {str(e)}")
        return pd.DataFrame()

def main():
    st.header("📈 Next Working Day Prediction Accuracy")
    
    st.info("Shows next working day predictions with both Workers (NoOfMan) and Hours.")
    
    # Get current and next working day
    current_date = datetime.now().date()
    next_date = get_next_working_day(current_date)
    
    if next_date is None:
        st.error("Could not determine next working day.")
        return
    
    # Display context
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Date", current_date.strftime("%Y-%m-%d (%A)"))
    with col2:
        st.metric("Predicting For", next_date.strftime("%Y-%m-%d (%A)"))
    
    # Load data
    prediction_df = load_prediction_data(next_date.strftime("%Y-%m-%d"))
    
    if prediction_df is None:
        st.warning("No original prediction data found.")
        return
    
    # Calculate improved predictions
    improved_workers, improved_hours = calculate_improved_prediction(prediction_df, next_date)
    
    # Create comparison
    comparison_df = create_comparison_dataframe(prediction_df, improved_workers, improved_hours, next_date)
    
    if not comparison_df.empty:
        st.subheader("Original vs. Improved Predictions")
        
        # Transpose the data to have Work Types as columns
        transposed_data = []
        
        # Get all punch codes
        punch_codes = comparison_df['PunchCode'].tolist()
        
        # Create rows for each metric
        metrics = [
            ('Original Workers', 'Original Workers'),
            ('Improved Workers', 'Improved Workers'),
            ('Workers Change', 'Workers Change'),
            ('Original Hours', 'Original Hours'),
            ('Improved Hours', 'Improved Hours'),
            ('Hours Change', 'Hours Change')
        ]
        
        for metric_name, column_name in metrics:
            row = {'Metric': metric_name}
            total_value = 0
            
            for punch_code in punch_codes:
                if punch_code != 'TOTAL':  # Skip if TOTAL already exists
                    value = comparison_df[comparison_df['PunchCode'] == punch_code][column_name].iloc[0]
                    row[punch_code] = value
                    total_value += value
            
            # Add TOTAL column
            row['TOTAL'] = total_value
            transposed_data.append(row)
        
        # Create transposed dataframe
        transposed_df = pd.DataFrame(transposed_data)
        
        # Get punch codes for column configuration (excluding Metric and TOTAL)
        punch_code_columns = [col for col in transposed_df.columns if col not in ['Metric', 'TOTAL']]
        
        # Create column configuration
        column_config = {
            'Metric': st.column_config.TextColumn("Metric", width="medium")
        }
        
        # Add punch code columns
        for punch_code in punch_code_columns:
            column_config[punch_code] = st.column_config.NumberColumn(
                punch_code,
                format="%.1f" if 'Workers' in transposed_df[transposed_df['Metric'].str.contains('Workers')]['Metric'].values else "%.0f"
            )
        
        # Add TOTAL column with special formatting
        column_config['TOTAL'] = st.column_config.NumberColumn(
            "TOTAL",
            format="%.1f",
            help="Total across all punch codes"
        )
        
        # Display the transposed table
        st.dataframe(
            transposed_df,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
        
        
        # Summary metrics - extract totals from transposed data
        st.subheader("Summary")
        
        # Extract values safely
        original_workers = transposed_df[transposed_df['Metric'] == 'Original Workers']['TOTAL'].iloc[0]
        improved_workers = transposed_df[transposed_df['Metric'] == 'Improved Workers']['TOTAL'].iloc[0]
        workers_change = transposed_df[transposed_df['Metric'] == 'Workers Change']['TOTAL'].iloc[0]
        original_hours = transposed_df[transposed_df['Metric'] == 'Original Hours']['TOTAL'].iloc[0]
        improved_hours = transposed_df[transposed_df['Metric'] == 'Improved Hours']['TOTAL'].iloc[0]
        hours_change = transposed_df[transposed_df['Metric'] == 'Hours Change']['TOTAL'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Original Workers", f"{original_workers:.1f}")
        
        with col2:
            st.metric(
                "Total Improved Workers", 
                f"{improved_workers:.1f}",
                delta=f"{workers_change:.1f}"
            )
        
        with col3:
            st.metric("Total Original Hours", f"{original_hours:.0f}")
        
        with col4:
            st.metric(
                "Total Improved Hours", 
                f"{improved_hours:.0f}",
                delta=f"{hours_change:.0f}"
            )
        

    else:
        st.warning("No comparison data available.")

if __name__ == "__main__":
    main()
    StateManager.initialize()