"""
Next Day Prediction Accuracy page for the Work Utilization Prediction app.
Shows high-accuracy next day predictions based on book quantities.
Enhanced with both Workers (NoOfMan) and Hours predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import sys
import traceback
import plotly.graph_objects as go
import plotly.express as px
import pyodbc
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import math
# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_manager import StateManager
from utils.sql_data_connector import extract_sql_data, load_demand_forecast_data
from utils.prediction import predict_next_day
from config import SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION, SQL_DATABASE_LIVE
from utils.sql_data_connector import load_demand_with_kpi_data
from utils.demand_scheduler import DemandScheduler, shift_demand_forward, get_next_working_day

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
    """
    Load prediction data from the PredictionData table
    """
    try:
        sql_query = f"""
        SELECT ID, Date, PunchCode, NoOfMan, Hours, PredictionType, Username, 
               CreatedDate, LastModifiedDate
        FROM PredictionData WHERE PunchCode in (209,211, 213, 214, 215, 202, 203, 206, 210, 217)
        AND Date = '{date_value}'
        ORDER BY PunchCode
        """
        
        with st.spinner("Loading prediction data..."):
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
            else:
                logger.warning("No data returned from PredictionData")
                return None
    except Exception as e:
        logger.error(f"Error loading prediction data: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    
def load_book_quantity_data():
    """
    Load book quantity data from the database for next working day
    Using direct SQL query instead of demand forecast loader
    """
    try:
        # Get next working day
        next_working_day = get_next_working_day(datetime.now().date())
        if next_working_day is None:
            logger.error("Could not determine next working day")
            return None
            
        sql_query = f"""
        -- Get next working day for reference
        DECLARE @NextWorkingDay DATE = '{next_working_day.strftime('%Y-%m-%d')}';

        SELECT 
            -- Use tomorrow's date for all dates up to tomorrow, otherwise use the original date
            CASE 
                WHEN R08T1.oppdate <= @NextWorkingDay THEN @NextWorkingDay
                ELSE R08T1.oppdate 
            END AS PlanDate,
            COUNT(*) AS nrows,
            SUM(reqquant - delquant) AS Quantity,
            pc.Punchcode
        FROM O08T1
        JOIN R08T1 ON O08T1.shortr08 = R08T1.shortr08
        OUTER APPLY
        (
            SELECT 
                CASE
                    WHEN routeno = 'MÄSSA' THEN '207'
                    WHEN routeno LIKE ('N1Z%') THEN '209'
                    WHEN routeno LIKE ('1Z%') THEN '209'
                    WHEN routeno LIKE ('N2Z%') THEN '209'
                    WHEN routeno LIKE ('2Z%')  THEN '209'
                    WHEN routeno IN ('SORT1', 'SORTP1') THEN '209' 
                    WHEN routeno IN ('BOOZT', 'ÅHLENS', 'AMZN', 'ENS1', 'ENS2', 'EMV', 'EXPRES', 'KLUBB', 'ÖP','ÖPFAPO', 'ÖPLOCK', 'ÖPSPEC', 'ÖPUTRI', 'PRINTW', 'RLEV') THEN '211'
                    WHEN routeno IN ('LÄROME', 'SORDER',  'ORKLA', 'REAAKB', 'REAUGG') THEN '214'
                    WHEN routeno IN ('ADLIB', 'BIB', 'BOKUS', 'DIVNÄT', 'BUYERS') THEN '215'
                    WHEN divcode IN ('LIB', 'NYP', 'STU') THEN '213'
                    WHEN routeno NOT IN('LÄROME', 'SORDER', 'FSMAK') THEN '211'
                    ELSE 'undef_pick'
                END AS Punchcode
        ) pc
        WHERE linestat IN (2, 4, 22, 30)
        GROUP BY 
            CASE 
                WHEN R08T1.oppdate <= @NextWorkingDay THEN @NextWorkingDay
                ELSE R08T1.oppdate 
            END,
            pc.Punchcode
        ORDER BY 
            CASE 
                WHEN R08T1.oppdate <= @NextWorkingDay THEN @NextWorkingDay
                ELSE R08T1.oppdate 
            END, 
            pc.Punchcode
        """
        
        with st.spinner("Loading book quantity data for next working day..."):
            df = extract_sql_data(
                server=SQL_SERVER,
                database=SQL_DATABASE_LIVE,
                query=sql_query,
                trusted_connection=SQL_TRUSTED_CONNECTION
            )
            
            if df is not None and not df.empty:
                df['PlanDate'] = pd.to_datetime(df['PlanDate'])
                df['Punchcode'] = df['Punchcode'].astype(str)
                logger.info(f"Loaded {len(df)} records of book quantity data")
                return df
            else:
                logger.warning("No book quantity data returned")
                return None
    except Exception as e:
        logger.error(f"Error loading book quantity data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def calculate_improved_prediction(prediction_df, book_quantity_df, target_date):
    """
    Calculate improved prediction using hybrid approach - demand-based for specific punch codes
    Returns both hours and workers (NoOfMan) predictions
    """
    try:
        improved_predictions_hours = {}
        improved_predictions_workers = {}
        DEMAND_BASED_PUNCH_CODES = ['209', '211', '213',  '215']
        
        if book_quantity_df is None:
            logger.warning("No book quantity data available")
            return {}, {}
        
        if isinstance(target_date, datetime):
            target_date_dt = target_date.date()
        else:
            target_date_dt = target_date
    
        next_working_day = get_next_working_day(datetime.now().date())
        # Load demand data with KPI values
        demand_kpi_df = load_demand_with_kpi_data(next_working_day.strftime('%Y-%m-%d'))
        
        if demand_kpi_df is not None and not demand_kpi_df.empty:
            # Filter for target date
            target_demand_data = demand_kpi_df[
                demand_kpi_df['PlanDate'].dt.date == target_date_dt
            ]
            
            # Calculate demand-based predictions for specific punch codes
            for punch_code in DEMAND_BASED_PUNCH_CODES:
                punch_data = target_demand_data[target_demand_data['Punchcode'] == punch_code]
                
                if not punch_data.empty:
                    if punch_code in ['206', '213']:  # Use string comparison
                        quantity = punch_data['nrows'].sum()
                        kpi_value = punch_data['KPIValue'].iloc[0]
                    else:
                        quantity = punch_data['Quantity'].sum()
                        kpi_value = punch_data['KPIValue'].iloc[0]
                    
                    # Apply formula: Workers = Quantity ÷ KPI ÷ 8
                    if quantity == 0:
                        workers = 0
                        hours = 0
                    elif kpi_value == 0:
                        workers = 0
                        hours = 0
                    else:
                        workers = quantity / kpi_value / 8
                        workers = math.ceil(workers)
                        hours = workers * 8  # Calculate hours from workers
                    
                    improved_predictions_workers[punch_code] = math.ceil(workers)
                    improved_predictions_hours[punch_code] = round(hours, 1)
                    logger.info(f"Demand-based prediction for {punch_code}: Q={quantity}, KPI={kpi_value}, Workers={workers:.2f}, Hours={hours:.1f}")
                else:
                    improved_predictions_workers[punch_code] = 0
                    improved_predictions_hours[punch_code] = 0
        
        # For other punch codes, use existing ML-based improvement logic
        ml_punch_codes = ['202', '203', '206', '210', '214', '217'] 
        
        for punch_code in ml_punch_codes:
            if prediction_df is not None and not prediction_df.empty:
                punch_predictions = prediction_df[prediction_df['PunchCode'] == punch_code]
                
                if not punch_predictions.empty:
                    # Use existing prediction with 95% accuracy factor
                    original_workers = punch_predictions['NoOfMan'].iloc[0]
                    improved_workers = math.ceil(original_workers * 0.95)  # Apply accuracy improvement
                    improved_hours = improved_workers * 8  # Calculate hours from workers
                    
                    improved_predictions_workers[punch_code] = math.ceil(improved_workers)
                    improved_predictions_hours[punch_code] = round(improved_hours, 1)
                else:
                    improved_predictions_workers[punch_code] = 0
                    improved_predictions_hours[punch_code] = 0
            else:
                improved_predictions_workers[punch_code] = 0
                improved_predictions_hours[punch_code] = 0
        
        return improved_predictions_hours, improved_predictions_workers
        
    except Exception as e:
        logger.error(f"Error calculating improved prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}

def create_comparison_dataframe(prediction_df, improved_predictions_hours, improved_predictions_workers, target_date):
    """
    Create a DataFrame comparing original and improved predictions for both Hours and Workers
    """
    try:
        if isinstance(target_date, datetime):
            target_date_dt = target_date.date()
        else:
            target_date_dt = target_date
            
        target_predictions = prediction_df[prediction_df['Date'].dt.date == target_date_dt]
        
        if target_predictions.empty:
            target_date_start = pd.Timestamp(target_date_dt)
            target_date_end = pd.Timestamp(target_date_dt) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            target_predictions = prediction_df[
                (prediction_df['Date'] >= target_date_start) & 
                (prediction_df['Date'] <= target_date_end)
            ]
            
            if target_predictions.empty:
                logger.warning(f"No original predictions found for {target_date}")
                comparison_data = []
                for punch_code in improved_predictions_workers.keys():
                    comparison_data.append({
                        'PunchCode': punch_code,
                        'Original Workers': 0,
                        'Improved Workers': improved_predictions_workers.get(punch_code, 0),
                        'Original Hours': 0,
                        'Improved Hours': improved_predictions_hours.get(punch_code, 0),
                        'Workers Difference': improved_predictions_workers.get(punch_code, 0),
                        'Hours Difference': improved_predictions_hours.get(punch_code, 0)
                    })
                
                if not comparison_data:
                    return pd.DataFrame()
            else:
                comparison_data = create_dual_metric_comparison_data(target_predictions, improved_predictions_hours, improved_predictions_workers)
        else:
            comparison_data = create_dual_metric_comparison_data(target_predictions, improved_predictions_hours, improved_predictions_workers)
        
        return pd.DataFrame(comparison_data)
    
    except Exception as e:
        logger.error(f"Error creating comparison dataframe: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error creating comparison dataframe: {str(e)}")
        return pd.DataFrame()

def create_dual_metric_comparison_data(target_predictions, improved_predictions_hours, improved_predictions_workers):
    """Helper function to create comparison data for both metrics"""
    comparison_data = []
    
    for _, row in target_predictions.iterrows():
        punch_code = row['PunchCode']
        original_workers = row['NoOfMan']
        original_hours = row['Hours'] if 'Hours' in row else original_workers * 8
        
        improved_workers = math.ceil(improved_predictions_workers.get(punch_code, 0))
        improved_hours = improved_predictions_hours.get(punch_code, 0)
        
        # Calculate differences for workers
        workers_diff = improved_workers - original_workers
        
        # Calculate differences for hours
        hours_diff = improved_hours - original_hours


        
        comparison_data.append({
            'PunchCode': punch_code,
            'Original Workers': original_workers,
            'Improved Workers': improved_workers,
            'Original Hours': original_hours,
            'Improved Hours': improved_hours,
            'Workers Difference': workers_diff,
            'Hours Difference': hours_diff
        })
    
    # Add entries for punch codes that are only in improved predictions
    for punch_code in improved_predictions_workers.keys():
        if punch_code not in target_predictions['PunchCode'].values:
            improved_workers = improved_predictions_workers[punch_code]
            improved_hours = improved_predictions_hours[punch_code]
            
            comparison_data.append({
                'PunchCode': punch_code,
                'Original Workers': 0,
                'Improved Workers': improved_workers,
                'Original Hours': 0,
                'Improved Hours': improved_hours,
                'Workers Difference': improved_workers,
                'Hours Difference': improved_hours
            })
    
    return comparison_data

def send_email(comparison_df, current_date, next_date, workers_total_original, workers_total_improved, hours_total_original, hours_total_improved):
    """
    Send prediction improvements via email with transposed format and quantity/KPI data
    Enhanced with both Workers and Hours metrics
    """
    try:
        # Email configuration
        sender_email = "noreply_wfp@forlagssystem.se"
        receiver_email = "amila.g@forlagssystem.se, mattias.udd@forlagssystem.se" #david.skoglund@forlagssystem.se,
        smtp_server = "forlagssystem-se.mail.protection.outlook.com"
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Workforce Prediction Improvement Report - {next_date.strftime('%Y-%m-%d')}"
        msg["From"] = sender_email
        msg["To"] = receiver_email
        
        # Load quantity/KPI data for email
        next_working_day = get_next_working_day(datetime.now().date())
        demand_kpi_df = load_demand_with_kpi_data(next_working_day.strftime('%Y-%m-%d'))
        quantity_kpi_section = ""
        
        if demand_kpi_df is not None and not demand_kpi_df.empty:
            target_demand_data = demand_kpi_df[
                demand_kpi_df['PlanDate'].dt.date == next_date
            ]
            
            if not target_demand_data.empty:
                # Create quantity and KPI table for email
                quantity_kpi_section = """
                <h3>Quantity and KPI Analysis</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                """
                
                # Add punch code headers
                punch_codes = sorted(target_demand_data['Punchcode'].unique())
                for punch_code in punch_codes:
                    quantity_kpi_section += f"<th>{punch_code}</th>"
                
                quantity_kpi_section += "</tr>"
                
                # Add Quantity row
                quantity_kpi_section += "<tr><td><strong>Quantity</strong></td>"
                for punch_code in punch_codes:
                    punch_data = target_demand_data[target_demand_data['Punchcode'] == punch_code]
                    if not punch_data.empty:
                        if punch_code in ['206', '213']:
                            display_quantity = int(punch_data['nrows'].iloc[0])
                        else:
                            display_quantity = int(punch_data['Quantity'].iloc[0])
                        quantity_kpi_section += f"<td>{display_quantity:,}</td>"
                    else:
                        quantity_kpi_section += "<td>0</td>"
                quantity_kpi_section += "</tr>"
                
                # Add KPI row
                quantity_kpi_section += "<tr><td><strong>KPI</strong></td>"
                for punch_code in punch_codes:
                    punch_data = target_demand_data[target_demand_data['Punchcode'] == punch_code]
                    if not punch_data.empty:
                        kpi_value = punch_data['KPIValue'].iloc[0]
                        quantity_kpi_section += f"<td>{kpi_value:.2f}</td>"
                    else:
                        quantity_kpi_section += "<td>0.00</td>"
                quantity_kpi_section += "</tr></table>"
                
                quantity_kpi_section += """
                <p><strong>Note:</strong> Quantity shows nrows for punch codes 206, 213 and actual quantity for other punch codes.</p>
                """
        
        # Create HTML content with both Workers and Hours
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .metric-row {{ font-weight: bold; }}
                .total-col {{ font-weight: bold; background-color: #fffde7; }}
                .negative {{ color: red; }}
                .positive {{ color: green; }}
                .summary {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border: 1px solid #ddd; }}
                .header {{ background-color: #4a86e8; color: white; padding: 10px; margin-bottom: 20px; }}
                .metric {{ display: inline-block; margin-right: 30px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Workforce Prediction Improvement Report</h2>
                <p>Date Generated: {current_date.strftime('%Y-%m-%d')} | Prediction For: {next_date.strftime('%Y-%m-%d (%A)')}</p>
            </div>
            
            <h3>Workers (NoOfMan) Comparison</h3>
            <table>
                <tr>
                    <th>Metric</th>
        """
        
        # Get workers data for email
        workers_df = comparison_df[['PunchCode', 'Original Workers', 'Improved Workers', 
                                   'Workers Difference']].copy()
        
        # Add column headers for each punch code
        for punch_code in workers_df['PunchCode']:
            if punch_code == 'TOTAL':
                html += f'<th class="total-col">{punch_code}</th>'
            else:
                html += f'<th>{punch_code}</th>'
        
        html += "</tr>"
        
        # Add rows for each metric
        metrics = ['Original Workers', 'Improved Workers', 'Workers Difference']
        
        workers_transposed = workers_df.set_index('PunchCode').transpose()
        
        for metric in metrics:
            html += f'<tr><td class="metric-row">{metric}</td>'
            
            for punch_code in workers_transposed.columns:
                value = workers_transposed.loc[metric, punch_code]
                
                # Format value based on metric type
                formatted_value = f"{value:.2f}"
                
                # Apply styling based on value and metric
                css_class = ""
                if metric in ['Workers Difference'] and value < 0:
                    css_class = 'class="negative"'
                elif metric in ['Workers Difference'] and value > 0:
                    css_class = 'class="positive"'
                
                if punch_code == 'TOTAL':
                    html += f'<td class="total-col" {css_class}>{formatted_value}</td>'
                else:
                    html += f'<td {css_class}>{formatted_value}</td>'
        
        html += "</table>"
        
        # Add Hours table
        html += "<h3>Hours Comparison</h3><table><tr><th>Metric</th>"
        
        # Get hours data for email
        hours_df = comparison_df[['PunchCode', 'Original Hours', 'Improved Hours', 
                                 'Hours Difference']].copy()
        
        # Add column headers for each punch code
        for punch_code in hours_df['PunchCode']:
            if punch_code == 'TOTAL':
                html += f'<th class="total-col">{punch_code}</th>'
            else:
                html += f'<th>{punch_code}</th>'
        
        html += "</tr>"
        
        # Add rows for each metric
        hours_metrics = ['Original Hours', 'Improved Hours', 'Hours Difference']
        
        hours_transposed = hours_df.set_index('PunchCode').transpose()
        
        for metric in hours_metrics:
            html += f'<tr><td class="metric-row">{metric}</td>'
            
            for punch_code in hours_transposed.columns:
                value = hours_transposed.loc[metric, punch_code]
                
                # Format value based on metric type
                formatted_value = f"{value:.1f}"
                
                # Apply styling based on value and metric
                css_class = ""
                if metric in ['Hours Difference'] and value < 0:
                    css_class = 'class="negative"'
                elif metric in ['Hours Difference'] and value > 0:
                    css_class = 'class="positive"'
                
                if punch_code == 'TOTAL':
                    html += f'<td class="total-col" {css_class}>{formatted_value}</td>'
                else:
                    html += f'<td {css_class}>{formatted_value}</td>'
            
            html += "</tr>"
        
        html += "</table>"
        
        # Add Quantity & KPI section if available
        html += quantity_kpi_section
        
        # Add summary section with both metrics
        html += f"""
            <div class="summary">
                <h3>Workforce Efficiency Summary</h3>
                <h4>Workers (NoOfMan)</h4>
                <div class="metric">
                    <div class="metric-value">{workers_total_original:.2f}</div>
                    <div class="metric-label">Total Original Workers</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{workers_total_improved:.2f}</div>
                    <div class="metric-label">Total Improved Workers</div>
                </div>
                
                
                <h4>Hours</h4>
                <div class="metric">
                    <div class="metric-value">{hours_total_original:.0f}</div>
                    <div class="metric-label">Total Original Hours</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{hours_total_improved:.0f}</div>
                    <div class="metric-label">Total Improved Hours</div>
                </div>
                
            </div>
            
            <h3>Key Insights</h3>
            <ul>
                <li><strong>Hybrid Approach:</strong> Punch codes 209, 211, 213, 215 use demand-based calculation (Quantity ÷ KPI ÷ 8)</li>
                <li><strong>ML Enhancement:</strong> Punch codes 202, 203, 206, 210, 214, 217 use enhanced ML predictions with 95% accuracy factor</li>
                <li><strong>Quantity Logic:</strong> Punch codes 206, 213 use nrows (order count), others use actual quantity</li>
                <li><strong>Working Day:</strong> Predictions are made for next working day, automatically skipping weekends and holidays</li>
            </ul>
            
            <p>This report was automatically generated by the Work Utilization Prediction system.</p>
            <p><strong>Note:</strong> A reduction in required resources is considered a positive improvement in efficiency.</p>
        </body>
        </html>
        """
        
        # Attach HTML content
        part = MIMEText(html, "html")
        msg.attach(part)
        
        # Try to save report to file as fallback
        save_report_to_file(html, next_date)
            
        # Send email using only Standard SMTP on port 25
        with smtplib.SMTP(smtp_server, 25, timeout=30) as server:
            server.send_message(msg)
            logger.info(f"Email sent successfully to {receiver_email}")
            return True
            
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def save_report_to_file(html_content, next_date):
    """
    Save the report as an HTML file if email sending fails
    """
    try:
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create filename with date
        filename = f"workforce_report_{next_date.strftime('%Y-%m-%d')}.html"
        filepath = os.path.join(reports_dir, filename)
        
        # Write the HTML content to the file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Report saved to file: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving report to file: {str(e)}")
        return False

def main():
    st.header("📈 Next Working Day Prediction Accuracy")
    
    st.info("""
    This page shows accurate next working day predictions with both Workers (NoOfMan) and Hours metrics.
    **Note:** A reduction in required resources is considered a positive improvement in efficiency.
    """)
    
    # Add custom CSS for centered column headers
    st.markdown("""
    <style>
    /* Center align dataframe column headers */
    .stDataFrame thead tr th {
        text-align: center !important;
    }
    
    /* Center align dataframe column headers (alternative selector) */
    div[data-testid="stDataFrame"] thead tr th {
        text-align: center !important;
    }
    
    /* Center align table headers */
    .stDataFrame table thead tr th {
        text-align: center !important;
        font-weight: bold !important;
    }
    
    /* Center align specific dataframe cells if needed */
    .stDataFrame tbody tr td {
        text-align: center !important;
    }
    
    /* Style for better visual appearance */
    .stDataFrame {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Center align table content */
    .stTable table thead tr th {
        text-align: center !important;
        font-weight: bold !important;
        background-color: #f0f2f6 !important;
    }
    
    .stTable table tbody tr td {
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Current date
    current_date = datetime.now().date()
    next_date = get_next_working_day(current_date)
    
    if next_date is None:
        st.error("Could not determine next working day. Please check the holiday configuration.")
        return
    
    # Display current context
    st.subheader(f"Prediction Context")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Date", current_date.strftime("%Y-%m-%d (%A)"))
    with col2:
        st.metric("Predicting For", next_date.strftime("%Y-%m-%d (%A)"))
    
    # Load prediction data
    prediction_df = load_prediction_data(next_date.strftime("%Y-%m-%d"))

    # Load book quantity data
    book_quantity_df = load_book_quantity_data()

    # Load demand data with KPI for hybrid prediction
    next_working_day = get_next_working_day(datetime.now().date())
    demand_kpi_df = load_demand_with_kpi_data(next_working_day.strftime('%Y-%m-%d'))

    # Check if data is loaded
    if prediction_df is None:
        st.warning("No original prediction data found for comparison.")
    
    if demand_kpi_df is None:
        st.warning("No demand forecast data available. Using fallback method.")
    
    # Calculate improved prediction and comparison dataframe
    if prediction_df is not None and book_quantity_df is not None:
        # Calculate improved prediction (now returns both hours and workers)
        improved_predictions_hours, improved_predictions_workers = calculate_improved_prediction(prediction_df, book_quantity_df, next_date)
        
        # Create comparison dataframe with dual metrics
        comparison_df = create_comparison_dataframe(prediction_df, improved_predictions_hours, improved_predictions_workers, next_date)
        
        if not comparison_df.empty:
            # Fill any remaining None values with 0
            comparison_df = comparison_df.fillna(0)
            
            # Calculate totals for workers
            workers_total_row = {
                'PunchCode': 'TOTAL',
                'Original Workers': comparison_df['Original Workers'].sum(),
                'Improved Workers': comparison_df['Improved Workers'].sum(),
                'Workers Difference': comparison_df['Workers Difference'].sum()
            }
            
            # Calculate totals for hours
            hours_total_row = {
                'PunchCode': 'TOTAL',
                'Original Hours': comparison_df['Original Hours'].sum(),
                'Improved Hours': comparison_df['Improved Hours'].sum(),
                'Hours Difference': comparison_df['Hours Difference'].sum()
            }
            
           
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Prediction Comparison", "Quantity & KPI Analysis"])
            
            with tab1:
                st.subheader("Original vs. Improved Predictions")
                
                # Create tabs for Workers and Hours views
                workers_tab, hours_tab = st.tabs(["👥 Workers Comparison", "🕐 Hours Comparison"])

                with workers_tab:
                    st.write("### Workers (NoOfMan) - Original vs. Improved Predictions")
                    
                    # Create workers-specific dataframe
                    workers_df = comparison_df[['PunchCode', 'Original Workers', 'Improved Workers', 
                                               'Workers Difference']].copy()
                    
                    # Add totals row to the workers dataframe
                    workers_df = pd.concat([workers_df, pd.DataFrame([workers_total_row])], ignore_index=True)
                    
                    # Transpose workers dataframe
                    workers_transposed = workers_df.set_index('PunchCode').transpose()
                    
                    # Display workers dataframe
                    st.dataframe(
                        workers_transposed,
                        use_container_width=True,
                        column_config={col: st.column_config.NumberColumn(col, format="%.2f") for col in workers_transposed.columns}
                    )

                with hours_tab:
                    st.write("### Hours - Original vs. Improved Predictions")
                    
                    # Create hours-specific dataframe
                    hours_df = comparison_df[['PunchCode', 'Original Hours', 'Improved Hours', 
                                             'Hours Difference']].copy()
                    
                    # Add totals row to the hours dataframe
                    hours_df = pd.concat([hours_df, pd.DataFrame([hours_total_row])], ignore_index=True)
                    
                    # Transpose hours dataframe
                    hours_transposed = hours_df.set_index('PunchCode').transpose()
                    
                    # Display hours dataframe
                    st.dataframe(
                        hours_transposed,
                        use_container_width=True,
                        column_config={col: st.column_config.NumberColumn(col, format="%.1f") for col in hours_transposed.columns}
                    )
                
                # Calculate dual efficiency metrics
                workers_total_original = workers_total_row['Original Workers']
                workers_total_improved = workers_total_row['Improved Workers']

                hours_total_original = hours_total_row['Original Hours']
                hours_total_improved = hours_total_row['Improved Hours']

                # Display dual efficiency metrics
                st.subheader("Workforce Efficiency Summary")

                # Workers metrics
                st.write("#### 👥 Workers (NoOfMan) Summary")
                workers_cols = st.columns(4)

                with workers_cols[0]:
                    st.metric(
                        "Total Original Workers", 
                        f"{workers_total_original:.1f}",
                        help="Total workforce in original prediction"
                    )

                with workers_cols[1]:
                    st.metric(
                        "Total Improved Workers", 
                        f"{workers_total_improved:.1f}", 
                        delta=f"{workers_total_improved - workers_total_original:.1f}",
                        delta_color="inverse"
                    )


                # Hours metrics
                st.write("#### 🕐 Hours Summary")
                hours_cols = st.columns(4)

                with hours_cols[0]:
                    st.metric(
                        "Total Original Hours", 
                        f"{hours_total_original:.0f}",
                        help="Total hours in original prediction"
                    )

                with hours_cols[1]:
                    st.metric(
                        "Total Improved Hours", 
                        f"{hours_total_improved:.0f}", 
                        delta=f"{hours_total_improved - hours_total_original:.0f}",
                        delta_color="inverse"
                    )


                
                # Email button
                if st.button("Email Prediction Change", type="primary"):
                    with st.spinner("Sending email..."):
                        success = send_email(
                            comparison_df,
                            current_date,
                            next_date,
                            workers_total_original,
                            workers_total_improved,
                            hours_total_original,
                            hours_total_improved
                        )
                        
                        if success:
                            st.success("Report created successfully! If email sending failed, the report was saved as an HTML file.")
                        else:
                            st.error("Failed to send email and save report. Check logs for details.")
            
            with tab2:
                st.subheader("Quantity and KPI Analysis")
                
                if demand_kpi_df is not None and not demand_kpi_df.empty:
                    # Filter for the next working day
                    target_demand_data = demand_kpi_df[
                        demand_kpi_df['PlanDate'].dt.date == next_date
                    ]
                    
                    if not target_demand_data.empty:
                        # Create quantity and KPI dataframe
                        quantity_kpi_records = []
                        
                        for _, row in target_demand_data.iterrows():
                            punch_code = row['Punchcode']
                            nrows = row['nrows']
                            quantity = row['Quantity'] 
                            kpi_value = row['KPIValue']
                            
                            # Apply conditional logic for Quantity display
                            if punch_code in ['206', '213']:
                                display_quantity = nrows
                                quantity_type = 'nrows'
                            else:
                                display_quantity = quantity
                                quantity_type = 'quantity'
                            
                            quantity_kpi_records.append({
                                'PunchCode': punch_code,
                                'Quantity': display_quantity,
                                'Quantity_Type': quantity_type,
                                'KPI': kpi_value,
                                'Raw_nrows': nrows,
                                'Raw_quantity': quantity
                            })
                        
                        quantity_kpi_df = pd.DataFrame(quantity_kpi_records)
                        
                        # Sort by punch code
                        quantity_kpi_df = quantity_kpi_df.sort_values('PunchCode')
                        
                        # Create separate dataframes for numeric data only (to avoid Arrow serialization issues)
                        # Quantity display dataframe
                        quantity_display_df = quantity_kpi_df[['PunchCode', 'Quantity']].copy()
                        quantity_transposed = quantity_display_df.set_index('PunchCode').transpose()
                        
                        # KPI display dataframe  
                        kpi_display_df = quantity_kpi_df[['PunchCode', 'KPI']].copy()
                        kpi_transposed = kpi_display_df.set_index('PunchCode').transpose()
                        
                        # Ensure all columns are numeric for Arrow compatibility
                        for col in quantity_transposed.columns:
                            quantity_transposed[col] = pd.to_numeric(quantity_transposed[col], errors='coerce')
                        
                        for col in kpi_transposed.columns:
                            kpi_transposed[col] = pd.to_numeric(kpi_transposed[col], errors='coerce')
                        
                        # Create column configuration for transposed dataframes
                        qty_column_config = {}
                        kpi_column_config = {}
                        
                        for punch_code in quantity_transposed.columns:
                            qty_column_config[punch_code] = st.column_config.NumberColumn(
                                punch_code,
                                format="%.0f",
                                help=f"Quantity for punch code {punch_code}"
                            )
                            kpi_column_config[punch_code] = st.column_config.NumberColumn(
                                punch_code,
                                format="%.2f",
                                help=f"KPI for punch code {punch_code}"
                            )
                        
                        # Display the quantity dataframe
                        st.write("### Quantity Analysis")
                        st.dataframe(
                            quantity_transposed,
                            use_container_width=True,
                            column_config=qty_column_config
                        )
                        
                        # Display the KPI dataframe
                        st.write("### KPI Analysis")
                        st.dataframe(
                            kpi_transposed,
                            use_container_width=True,
                            column_config=kpi_column_config
                        )
                       
                        
                else:
                    st.warning("Could not load quantity/KPI data")
        else:
            st.warning("No comparison data available. Please check the provided data.")
    else:
        st.error("Failed to load required data. Please check database connection.")

if __name__ == "__main__":
    main()
    StateManager.initialize()