"""
Data loading and manipulation utilities.
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
import streamlit as st
from datetime import datetime, timedelta
import traceback
from config import MODELS_DIR, DATA_DIR, CACHE_TTL, CHUNK_SIZE

# Configure logger
logger = logging.getLogger(__name__)

@st.cache_data(ttl=CACHE_TTL)
def load_data(file_path):
    """
    Load and preprocess the work_utilization_melted.xlsx file
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        # Check if it's a large file
        try:
            file_size = os.path.getsize(file_path) if isinstance(file_path, str) else file_path.size
            is_large_file = file_size > 100 * 1024 * 1024  # 100 MB
        except:
            is_large_file = False
        
        if is_large_file:
            logger.info("Large file detected. Loading in chunks.")
            # For large files, use a chunked approach
            with pd.ExcelFile(file_path) as xls:
                df_chunks = []
                for chunk in pd.read_excel(xls, sheet_name=0, chunksize=CHUNK_SIZE):
                    df_chunks.append(chunk)
                df = pd.concat(df_chunks)
        else:
            # For normal files, load directly
            df = pd.read_excel(file_path)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Ensure WorkType is treated as string
        df['WorkType'] = df['WorkType'].astype(str)
        
        # Fix Hours and NoOfMan columns - replace "-" with 0 and convert to numeric
        if 'Hours' in df.columns:
            df['Hours'] = df['Hours'].replace('-', 0)
            df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce').fillna(0)
            
        if 'NoOfMan' in df.columns:
            df['NoOfMan'] = df['NoOfMan'].replace('-', 0)
            df['NoOfMan'] = pd.to_numeric(df['NoOfMan'], errors='coerce').fillna(0)
        
        # Sort by Date
        df = df.sort_values('Date')
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values found in the dataset: {missing_values[missing_values > 0]}")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to load data: {str(e)}")

@st.cache_resource
def load_models():
    """
    Load saved prediction models, feature importances, and metrics
    
    Returns:
    --------
    tuple
        (models_dict, feature_importances_dict, metrics_dict)
    """
    try:
        models_path = os.path.join(MODELS_DIR, 'work_utilization_models.pkl')
        feature_importances_path = os.path.join(MODELS_DIR, 'work_utilization_feature_importances.pkl')
        metrics_path = os.path.join(MODELS_DIR, 'work_utilization_metrics.pkl')
        
        # Check if all files exist
        if not (os.path.exists(models_path) and 
                os.path.exists(feature_importances_path) and 
                os.path.exists(metrics_path)):
            logger.warning("One or more model files not found")
            return {}, {}, {}
        
        # Load models
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        
        # Load feature importances
        with open(feature_importances_path, 'rb') as f:
            feature_importances = pickle.load(f)
        
        # Load metrics
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        
        logger.info(f"Models loaded successfully. Number of models: {len(models)}")
        return models, feature_importances, metrics
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}, {}

def save_models(models, feature_importances, metrics):
    """
    Save trained models, feature importances, and metrics
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models for each WorkType
    feature_importances : dict
        Dictionary of feature importances for each WorkType
    metrics : dict
        Dictionary of evaluation metrics for each WorkType
    """
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save models
        with open(os.path.join(MODELS_DIR, 'work_utilization_models.pkl'), 'wb') as f:
            pickle.dump(models, f)
        
        # Save feature importances
        with open(os.path.join(MODELS_DIR, 'work_utilization_feature_importances.pkl'), 'wb') as f:
            pickle.dump(feature_importances, f)
        
        # Save metrics
        with open(os.path.join(MODELS_DIR, 'work_utilization_metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
        
        logger.info(f"Models saved successfully. Number of models: {len(models)}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
def export_predictions(predictions, file_path):
    """
    Export predictions to Excel or CSV
    
    Parameters:
    -----------
    predictions : dict
        Dictionary of predictions with dates as keys and WorkType predictions as values
    file_path : str
        Path to save the exported file
    """
    try:
        # Convert predictions to DataFrame
        rows = []
        for date, work_type_predictions in predictions.items():
            for work_type, value in work_type_predictions.items():
                rows.append({
                    'Date': date,
                    'WorkType': work_type,
                    'PredictedNoOfMan': value
                })
        
        df_predictions = pd.DataFrame(rows)
        
        # Save based on file extension
        if file_path.endswith('.xlsx'):
            df_predictions.to_excel(file_path, index=False)
        elif file_path.endswith('.csv'):
            df_predictions.to_csv(file_path, index=False)
        else:
            df_predictions.to_csv(file_path + '.csv', index=False)
        
        logger.info(f"Predictions exported successfully to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error exporting predictions: {str(e)}")
        logger.error(traceback.format_exc())
        return False