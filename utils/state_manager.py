# utils/state_manager.py
import streamlit as st

class StateManager:
    @staticmethod
    def initialize():
        """Initialize all required session state variables"""
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'models' not in st.session_state:
            st.session_state.models = None
        # etc.
    
    @staticmethod
    def set_data(df):
        """Set data with validation"""
        st.session_state.df = df
        return True
    
    @staticmethod
    def get_data():
        """Get data with appropriate fallbacks"""
        return st.session_state.df