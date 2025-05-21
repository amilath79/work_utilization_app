"""
Error reporting utilities for the Work Utilization Prediction app.
"""
import streamlit as st
import logging
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

def add_report_error_widget():
    """
    Add a small error reporting widget to the current page
    
    This widget provides a quick way for users to report errors from any page.
    """
    # Create expandable section at the bottom of the page
    with st.expander("⚠️ Report an Error or Issue", expanded=False):
        st.write("If you're experiencing any issues with this page, please use our error reporting tool:")
        
        # Button to navigate to the error reporting page
        if st.button("Report an Error", key="report_error_button"):
            # Log that someone clicked the button
            logger.info(f"Error report button clicked on page at {datetime.now()}")
            # Redirect to the error reporting page - using Streamlit's experimental navigation
            st.switch_page("pages/7_Error_Reporting.py")
            
        st.markdown("---")
        st.write("For immediate assistance, please contact support at support@example.com")

def add_floating_report_button():
    """
    Add a floating report error button using custom HTML/CSS
    
    Note: This uses custom components and may not work in all Streamlit versions.
    """
    # HTML/CSS for a floating button
    html_code = """
    <style>
        .floating-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #f44336;
            color: white;
            padding: 10px 15px;
            border-radius: 30px;
            text-decoration: none;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            font-weight: bold;
            z-index: 9999;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .floating-button:hover {
            background-color: #d32f2f;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        }
        .bug-icon {
            font-size: 18px;
        }
    </style>
    
    <a href="#" onclick="location.href='/7_Error_Reporting'" class="floating-button">
        <span class="bug-icon">🐛</span> Report Issue
    </a>
    """
    
    # Display the HTML
    st.markdown(html_code, unsafe_allow_html=True)

def init_error_tracking():
    """
    Initialize error tracking - call this at the start of each page
    
    This adds basic error tracking and reporting capabilities.
    """
    # Track that the page was visited
    current_page = st.experimental_get_query_params().get("page", ["unknown"])[0]
    logger.info(f"Page visited: {current_page}")
    
    # Set up exception handling for this page
    try:
        # Here we can setup global exception tracking if needed
        pass
    except Exception as e:
        logger.error(f"Error initializing error tracking: {str(e)}")