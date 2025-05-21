"""
Error Reporting page for the Work Utilization Prediction app.
"""
import streamlit as st
import pandas as pd
import logging
from datetime import datetime
import os
import sys
import json
import uuid

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.state_manager import StateManager
from config import DATA_DIR

# Configure page
st.set_page_config(
    page_title="Error Reporting",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logger
logger = logging.getLogger(__name__)

# Create directory for error reports if it doesn't exist
ERROR_REPORTS_DIR = os.path.join(DATA_DIR, "error_reports")
os.makedirs(ERROR_REPORTS_DIR, exist_ok=True)

def save_error_report(report_data):
    """
    Save error report to a JSON file
    
    Parameters:
    -----------
    report_data : dict
        Dictionary containing error report data
    
    Returns:
    --------
    str
        Path to the saved report file
    """
    try:
        # Generate a unique report ID
        report_id = str(uuid.uuid4())
        report_data['report_id'] = report_id
        
        # Add timestamp
        report_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create file path
        file_name = f"error_report_{report_id}.json"
        file_path = os.path.join(ERROR_REPORTS_DIR, file_name)
        
        # Save report to JSON file
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        logger.info(f"Error report saved to {file_path}")
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return None

def load_existing_reports():
    """
    Load existing error reports
    
    Returns:
    --------
    list
        List of error report dictionaries
    """
    reports = []
    try:
        # List all JSON files in the error reports directory
        for file_name in os.listdir(ERROR_REPORTS_DIR):
            if file_name.endswith('.json'):
                file_path = os.path.join(ERROR_REPORTS_DIR, file_name)
                
                # Load report from JSON file
                with open(file_path, 'r') as f:
                    report = json.load(f)
                
                reports.append(report)
        
        # Sort reports by timestamp (newest first)
        reports.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return reports
    
    except Exception as e:
        logger.error(f"Error loading reports: {str(e)}")
        return []

def main():
    st.header("🐛 Error Reporting")
    
    st.info("""
    If you encounter any issues or errors while using the application, please use this form to report them.
    Your feedback helps us improve the application.
    """)
    
    # Create tabs for reporting an error vs viewing existing reports
    tab1, tab2 = st.tabs(["Report an Error", "View Previous Reports"])
    
    with tab1:
        # Error reporting form
        st.subheader("Submit Error Report")
        
        with st.form("error_report_form"):
            # Error details
            error_type = st.selectbox(
                "Error Type",
                options=[
                    "Application Crash",
                    "Incorrect Data",
                    "Feature Not Working",
                    "Performance Issue",
                    "User Interface Problem",
                    "Other"
                ],
                help="Select the type of error you encountered"
            )
            
            # Page where error occurred
            page_options = [
                "Home/Dashboard",
                "Data Overview",
                "Predictions",
                "Model Analysis",
                "Backtesting",
                "Non-Working Days/Holidays",
                "KPI Management",
                "Demand Management",
                "Other"
            ]
            
            page = st.selectbox(
                "Where did the error occur?",
                options=page_options,
                help="Select the page where you encountered the error"
            )
            
            # Error description
            description = st.text_area(
                "Error Description",
                placeholder="Please describe the error in detail...",
                height=100,
                help="Provide details about what happened"
            )
            
            # Steps to reproduce
            steps = st.text_area(
                "Steps to Reproduce",
                placeholder="What steps can we follow to reproduce the error?",
                height=100,
                help="List the steps that led to the error"
            )
            
            # Error message (if any)
            error_message = st.text_input(
                "Error Message",
                placeholder="Copy and paste any error message you saw (if applicable)",
                help="If you saw an error message, please copy it here"
            )
            
            # Contact information (optional)
            st.write("#### Contact Information (Optional)")
            name = st.text_input("Name")
            email = st.text_input("Email")
            
            # Submit button
            submit_button = st.form_submit_button("Submit Report")
            
            if submit_button:
                # Validate required fields
                if not description:
                    st.error("Error description is required.")
                else:
                    # Create report data
                    report_data = {
                        "error_type": error_type,
                        "page": page,
                        "description": description,
                        "steps": steps,
                        "error_message": error_message,
                        "name": name,
                        "email": email
                    }
                    
                    # Save report
                    file_path = save_error_report(report_data)
                    
                    if file_path:
                        st.success("Error report submitted successfully! Thank you for your feedback.")
                        st.write(f"Report ID: {report_data['report_id']}")
                    else:
                        st.error("Failed to save error report. Please try again.")
    
    with tab2:
        # Display existing reports (only if there are any)
        reports = load_existing_reports()
        
        if reports:
            st.subheader(f"Previous Reports ({len(reports)})")
            
            # Create an expander for each report
            for i, report in enumerate(reports):
                with st.expander(f"Report #{i+1} - {report.get('error_type', 'Error')} - {report.get('timestamp', 'Unknown date')}"):
                    # Display report details
                    st.write(f"**Report ID:** {report.get('report_id', 'N/A')}")
                    st.write(f"**Date/Time:** {report.get('timestamp', 'N/A')}")
                    st.write(f"**Error Type:** {report.get('error_type', 'N/A')}")
                    st.write(f"**Page:** {report.get('page', 'N/A')}")
                    
                    st.write("**Description:**")
                    st.info(report.get('description', 'No description provided.'))
                    
                    st.write("**Steps to Reproduce:**")
                    st.info(report.get('steps', 'No steps provided.'))
                    
                    if report.get('error_message'):
                        st.write("**Error Message:**")
                        st.code(report.get('error_message'))
                    
                    if report.get('name') or report.get('email'):
                        st.write("**Contact Information:**")
                        if report.get('name'):
                            st.write(f"Name: {report.get('name')}")
                        if report.get('email'):
                            st.write(f"Email: {report.get('email')}")
        else:
            st.info("No error reports have been submitted yet.")
    
    # Help information
    with st.expander("Need more help?"):
        st.write("""
        If you need immediate assistance, please contact support at:
        
        - Email: support@example.com
        - Phone: +1-234-567-8900
        """)
        
        st.write("### Frequently Asked Issues")
        st.write("""
        1. **Database Connection Issues**: Check your SQL Server connection settings in the config.py file.
        2. **Missing Models**: Ensure that you have trained models by running the train_models.py script.
        3. **Visualization Errors**: Make sure you have the latest version of Plotly installed.
        """)

if __name__ == "__main__":
    main()
    StateManager.initialize()