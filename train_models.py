"""
Module for training workforce prediction models with productivity metrics.
For All PunchCodes
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import logging
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Import feature engineering functions from utils
from utils.feature_engineering import engineer_features, create_lag_features, get_feature_lists

from config import (
    MODELS_DIR, DATA_DIR, LAG_DAYS, ROLLING_WINDOWS, 
    CHUNK_SIZE, DEFAULT_MODEL_PARAMS,
    SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION,
    SQL_USERNAME, SQL_PASSWORD
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_models")

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(file_path):
    """
    Load and preprocess the work utilization data
    
    Parameters:
    -----------
    file_path : str
        Path to the data file (Excel or CSV)
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame
    """
    try:
        logger.info(f"Loading data from {file_path}")
        
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use Excel (.xlsx/.xls) or CSV (.csv)")
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Ensure Date column is datetime 
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle PunchCode as WorkType if it exists
        if 'PunchCode' in df.columns and 'WorkType' not in df.columns:
            df = df.rename(columns={'PunchCode': 'WorkType'})
            logger.info("Renamed 'PunchCode' column to 'WorkType'")
        
        # Ensure WorkType is treated as string
        df['WorkType'] = df['WorkType'].astype(str)
        
        # Process all numeric columns
        numeric_columns = ['Hours', 'NoOfMan', 'SystemHours', 'Quantity', 'ResourceKPI', 'SystemKPI']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].replace('-', 0)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Sort by Date
        df = df.sort_values('Date')
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to load data: {str(e)}")

def build_models(processed_data, work_types=None, n_splits=5):
    """
    Build and train a model for each WorkType using time series cross-validation
    
    Parameters:
    -----------
    processed_data : pd.DataFrame
        DataFrame with all features
    work_types : list, optional
        List of work types to build models for
    n_splits : int
        Number of splits for time series cross-validation
        
    Returns:
    --------
    tuple
        (models, feature_importances, metrics)
    """
    try:
        # If work_types is not provided, get them from the data
        if work_types is None:
            work_types = sorted(processed_data['WorkType'].unique())
        
        logger.info(f"Building models for {len(work_types)} work types")
        
        models = {}
        feature_importances = {}
        metrics = {}
        
        # Get feature lists from utility function
        numeric_features, categorical_features = get_feature_lists(
            include_advanced_features=True, 
            include_productivity_metrics=True
        )
        
        # Function to calculate modified MAPE with minimum threshold
        def modified_mape(y_true, y_pred, epsilon=1.0):
            """Calculate MAPE with a minimum threshold to avoid division by zero"""
            denominator = np.maximum(np.abs(y_true), epsilon)
            return np.mean(np.abs(y_pred - y_true) / denominator) * 100
        
        # Log all features that will be used
        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        for work_type in work_types:
            logger.info(f"Building model for WorkType: {work_type}")
            
            # Filter data for this WorkType
            work_type_data = processed_data[processed_data['WorkType'] == work_type] 
            
            if len(work_type_data) < 30:  # Skip if not enough data
                logger.warning(f"Skipping {work_type}: Not enough data ({len(work_type_data)} records)")
                continue
            
            # Sort data by date to ensure time-based splitting works correctly
            work_type_data = work_type_data.sort_values('Date')

            work_type_data.to_excel('work_type_data_All.xlsx', index=False)
            
            # Check which features are available in the dataset
            available_numeric = [f for f in numeric_features if f in work_type_data.columns]
            
            logger.info(f"Using {len(available_numeric)} numeric features and {len(categorical_features)} categorical features")
            
            # Prepare features and target
            X = work_type_data[available_numeric + categorical_features]
            y = work_type_data['NoOfMan']

            work_type_data.to_excel('work_type_data_PRe.xlsx', index=False)
            
            # Define preprocessing with imputation for missing values
            from sklearn.impute import SimpleImputer
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', SimpleImputer(strategy='median'), available_numeric),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ]
            )

            # Define the model pipeline using DEFAULT_MODEL_PARAMS from config
            model_params = DEFAULT_MODEL_PARAMS.copy()

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(**model_params))
            ])
            
            # Initialize TimeSeriesSplit with n splits
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Initialize metrics lists
            mae_scores = []
            rmse_scores = []
            r2_scores = []
            mape_scores = []
            
            # Perform time series cross-validation  
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics  
                mae_scores.append(mean_absolute_error(y_test, y_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred))) 
                r2_scores.append(r2_score(y_test, y_pred))
                
                # Calculate modified MAPE
                mape = modified_mape(y_test, y_pred, epsilon=1.0)
                mape_scores.append(mape)
            
            # Train final model on all data
            pipeline.fit(X, y)
            models[work_type] = pipeline
            
            # Get feature importances from the final model
            model = pipeline.named_steps['model']
            
            # Get feature names after preprocessing
            # Numeric features (after imputation)
            num_feature_names = available_numeric
            
            # Categorical features (after one-hot encoding)
            ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
            cat_feature_names = []
            for i, feature in enumerate(categorical_features):
                categories = ohe.categories_[i]
                for category in categories:
                    cat_feature_names.append(f"{feature}_{category}")
            
            # Combine feature names
            all_feature_names = num_feature_names + cat_feature_names
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Create dictionary of feature importances
            feature_importances[work_type] = dict(zip(all_feature_names, importances))
            
            # Store average metrics
            metrics[work_type] = {
                'MAE': np.mean(mae_scores),
                'RMSE': np.mean(rmse_scores), 
                'R²': np.mean(r2_scores),
                'MAPE': np.mean(mape_scores)
            }
            
            logger.info(f"Model for {work_type} - MAE: {metrics[work_type]['MAE']:.4f}, RMSE: {metrics[work_type]['RMSE']:.4f}, R²: {metrics[work_type]['R²']:.4f}, MAPE: {metrics[work_type]['MAPE']:.2f}%")
            
            # Print top 10 most important features
            importances_dict = feature_importances[work_type]
            sorted_importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"Top 10 most important features for {work_type}:")
            for feature, importance in sorted_importances[:10]:
                logger.info(f"  {feature}: {importance:.4f}")
            
            # Also print individual fold scores for detailed analysis
            logger.info(f"Cross-validation details for {work_type}:")
            for i in range(len(mae_scores)):
                logger.info(f"  Fold {i+1}: MAE={mae_scores[i]:.4f}, RMSE={rmse_scores[i]:.4f}, R²={r2_scores[i]:.4f}, MAPE={mape_scores[i]:.2f}%")
        
        return models, feature_importances, metrics
        
    except Exception as e:
        logger.error(f"Error building models: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}, {}

def train_from_sql(connection_string=None, sql_query=None):
    """
    Train models using data from a SQL query
    
    
    Parameters:
    -----------
    connection_string : str, optional
        SQL Server connection string
    sql_query : str, optional
        SQL query to execute
        
    Returns:
    --------
    tuple
        (models, feature_importances, metrics)
    """
    try:
        import pyodbc
        
        # Use default connection string from config if not provided
        if connection_string is None:
            connection_string = (
                f"DRIVER={{SQL Server}};"
                f"SERVER={SQL_SERVER};"
                f"DATABASE={SQL_DATABASE};"
            )
            
            if SQL_TRUSTED_CONNECTION:
                connection_string += "Trusted_Connection=yes;"
            else:
                connection_string += f"UID={SQL_USERNAME};PWD={SQL_PASSWORD};"
        
        # Default query if none provided
        if sql_query is None:
            sql_query = """
                SELECT Date, PunchCode as WorkType, Hours, NoOfMan, SystemHours, Quantity, ResourceKPI, SystemKPI 
                FROM WorkUtilizationData 
                WHERE PunchCode IN (215, 209, 213, 211, 214, 202, 203, 206, 210, 217) 
                AND Hours > 0 
                AND NoOfMan > 0 
                AND SystemHours > 0 
                AND Quantity > 0
                ORDER BY Date
            """
        
        logger.info(f"Connecting to database {SQL_DATABASE} on server {SQL_SERVER}")
        conn = pyodbc.connect(connection_string)
        logger.info(f"Executing SQL query: {sql_query}")
        
        # Handle large datasets with chunking
        chunks = []
        for chunk in pd.read_sql(sql_query, conn, chunksize=CHUNK_SIZE):
            chunks.append(chunk)
            logger.info(f"Read chunk of {len(chunk)} rows")
        
        conn.close()
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Data loaded successfully. Total rows: {len(df)}")
        else:
            logger.warning("No data returned from query")
            return None, None, None
        
        # Handle PunchCode as WorkType
        if 'PunchCode' in df.columns and 'WorkType' not in df.columns:
            df = df.rename(columns={'PunchCode': 'WorkType'})
        
        # IMPORTANT: Ensure Date column is properly converted to datetime
        # SQL Server dates may not be automatically recognized by pandas
        if 'Date' in df.columns:
            logger.info("Converting Date column to datetime format")
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Check for null dates after conversion
            null_dates = df['Date'].isna().sum()
            if null_dates > 0:
                logger.warning(f"Found {null_dates} null dates after conversion. Removing these rows.")
                df = df.dropna(subset=['Date'])
        
        # Ensure WorkType is treated as string
        df['WorkType'] = df['WorkType'].astype(str)
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['NoOfMan', 'Hours', 'SystemHours', 'Quantity', 'ResourceKPI', 'SystemKPI']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        
        # Process data and train models using utility functions
        logger.info("Engineering features...")
        feature_df = engineer_features(df)

        logger.info("Creating lag features...")
        lag_features_df = create_lag_features(feature_df)
        
        work_types = lag_features_df['WorkType'].unique()

        logger.info(f"Found {len(work_types)} unique work types")
        
        models, feature_importances, metrics = build_models(lag_features_df, work_types, n_splits=5)
        
        # Save models
        save_models(models, feature_importances, metrics)
        
        return models, feature_importances, metrics
        
    except Exception as e:
        logger.error(f"Error in train_from_sql: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def save_models(models, feature_importances, metrics):
    """
    Save trained models, feature importances, and metrics
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    feature_importances : dict
        Dictionary of feature importances
    metrics : dict
        Dictionary of model metrics
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Ensure models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        logger.info(f"Saving {len(models)} models to {MODELS_DIR}")
        
        # Save model files
        with open(os.path.join(MODELS_DIR, "work_utilization_models.pkl"), "wb") as f:
            pickle.dump(models, f)
            
        with open(os.path.join(MODELS_DIR, "work_utilization_feature_importances.pkl"), "wb") as f:
            pickle.dump(feature_importances, f)
            
        with open(os.path.join(MODELS_DIR, "work_utilization_metrics.pkl"), "wb") as f:
            pickle.dump(metrics, f)
        
        # Create a summary of model performance
        performance_summary = []
        for work_type, metric in metrics.items():
            performance_summary.append({
                'WorkType': work_type,
                'MAE': metric['MAE'],
                'RMSE': metric['RMSE'],
                'R²': metric['R²'],
                'MAPE': metric['MAPE']
            })
        
        performance_df = pd.DataFrame(performance_summary)
        performance_file = os.path.join(MODELS_DIR, "model_performance_summary.xlsx")
        performance_df.to_excel(performance_file, index=False)
        
        logger.info(f"Model files saved successfully")
        logger.info(f"Performance summary saved to {performance_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to run the training process"""
    try:
        logger.info("Starting the model training process")
        
        # Check if command line arguments were provided
        import sys
        use_sql = False
        
        if len(sys.argv) > 1:
            if sys.argv[1].lower() == 'sql':
                use_sql = True
                logger.info("Training from SQL database (command line argument)")
        
        if use_sql:
            # Train using SQL data
            result = train_from_sql()
            if result[0] is None:
                logger.error("SQL training failed. Check logs for details.")
                return False
        else:
            # Train using Excel file
            file_path = os.path.join(DATA_DIR, "work_utilization_melted1.csv")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                # Ask user for file path
                file_path = input("Enter path to data file (.xlsx or .csv): ")
            
            # Load the data
            df = load_data(file_path)
            
            # Use the feature engineering utilities 
            logger.info("Engineering features...")
            feature_df = engineer_features(df)
            
            logger.info("Creating lag features...")
            lag_features_df = create_lag_features(feature_df)
            
            # Get unique work types
            work_types = lag_features_df['WorkType'].unique()
            logger.info(f"Found {len(work_types)} unique work types")
            
            # Build and save models
            models, feature_importances, metrics = build_models(lag_features_df, work_types, n_splits=5)
            save_models(models, feature_importances, metrics)
        
        logger.info("Model training completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in main training process: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    main()