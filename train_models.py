"""
Module for training workforce prediction models with tiered feature system.
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
from utils.feature_engineering import engineer_features, create_lag_features
from utils.feature_builder import FeatureBuilder

from config import (
    MODELS_DIR, DATA_DIR, LAG_DAYS, ROLLING_WINDOWS, 
    CHUNK_SIZE, DEFAULT_MODEL_PARAMS,
    SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION,
    SQL_USERNAME, SQL_PASSWORD,
    FEATURE_TIERS, BASIC_FEATURES, INTERMEDIATE_FEATURES, ADVANCED_FEATURES
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
    Build and train a model for each WorkType using time series cross-validation with tiered features
    
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
        
        # Log which feature tiers are enabled
        active_tiers = [tier for tier, enabled in FEATURE_TIERS.items() if enabled]
        logger.info(f"Using feature tiers: {active_tiers}")
        
        models = {}
        feature_importances = {}
        metrics = {}
        
        # Use tiered feature system to get appropriate features
        feature_builder = FeatureBuilder(
            data_columns=list(processed_data.columns),
            data_length=len(processed_data)
        )
        
        # Get feature lists based on tier configuration
        numeric_features, categorical_features = feature_builder.get_feature_lists()
        
        # Function to calculate modified MAPE with minimum threshold
        def modified_mape(y_true, y_pred, epsilon=1.0):
            """Calculate MAPE with a minimum threshold to avoid division by zero"""
            denominator = np.maximum(np.abs(y_true), epsilon)
            return np.mean(np.abs(y_pred - y_true) / denominator) * 100
        
        # Log all features that will be used
        logger.info(f"Tiered feature system generated {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")
        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        for work_type in work_types:
            try:
                logger.info(f"Building model for WorkType: {work_type}")
                avg_value = diagnose_training_data(processed_data, work_type)
                print(f'Work Type : {work_type} - Avarage Value : {avg_value}')
                # Filter data for this WorkType
                work_type_data = processed_data[processed_data['WorkType'] == work_type] 
                
                if len(work_type_data) < 30:  # Skip if not enough data
                    logger.warning(f"Skipping {work_type}: Not enough data ({len(work_type_data)} records)")
                    continue
                
                # Sort data by date to ensure time-based splitting works correctly
                work_type_data = work_type_data.sort_values('Date')
                
                # Check which features are available in the dataset
                available_numeric = [f for f in numeric_features if f in work_type_data.columns]
                available_categorical = [f for f in categorical_features if f in work_type_data.columns]
                
                logger.info(f"Available features for {work_type}: {len(available_numeric)} numeric, {len(available_categorical)} categorical")
                
                # Debug: Log missing features if many are missing
                missing_numeric = [f for f in numeric_features if f not in work_type_data.columns]
                missing_categorical = [f for f in categorical_features if f not in work_type_data.columns]
                
                if missing_numeric:
                    logger.debug(f"Missing numeric features for {work_type}: {missing_numeric}")
                if missing_categorical:
                    logger.debug(f"Missing categorical features for {work_type}: {missing_categorical}")
                
                # Skip if no features are available
                if len(available_numeric) == 0 and len(available_categorical) == 0:
                    logger.warning(f"Skipping {work_type}: No features available")
                    continue
                
                # Prepare features and target
                all_available_features = available_numeric + available_categorical
                X = work_type_data[all_available_features]
                y = work_type_data['NoOfMan']
                
                # Define preprocessing with imputation for missing values
                from sklearn.impute import SimpleImputer
                
                transformers = []
                if available_numeric:
                    transformers.append(('num', SimpleImputer(strategy='median'), available_numeric))
                if available_categorical:
                    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), available_categorical))
                
                preprocessor = ColumnTransformer(transformers=transformers)

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
                    

                    bias_ratio = validate_model_performance(pipeline, X, y, work_type)
                    print(f'Basic Ratio for {work_type} : {bias_ratio}')
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
                feature_names = []
                
                # Add numeric feature names (they stay the same after imputation)
                if available_numeric:
                    feature_names.extend(available_numeric)
                
                # Add categorical feature names (expanded after one-hot encoding)
                if available_categorical:
                    preprocessor_fitted = pipeline.named_steps['preprocessor']
                    
                    # Check if categorical transformer exists by looking at transformer names
                    transformer_names = [name for name, transformer, columns in preprocessor_fitted.transformers_]
                    
                    if 'cat' in transformer_names:
                        try:
                            ohe = preprocessor_fitted.named_transformers_['cat']
                            for i, feature in enumerate(available_categorical):
                                categories = ohe.categories_[i]
                                for category in categories:
                                    feature_names.append(f"{feature}_{category}")
                        except Exception as cat_error:
                            logger.warning(f"Error processing categorical features for {work_type}: {str(cat_error)}")
                            # Fallback: just use original categorical feature names
                            feature_names.extend(available_categorical)
                
                # Get feature importances
                importances = model.feature_importances_
                
                # Validate that we have the right number of feature names
                if len(feature_names) != len(importances):
                    logger.warning(f"Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)}) for {work_type}")
                    # Create generic feature names as fallback
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                # Create dictionary of feature importances
                feature_importances[work_type] = dict(zip(feature_names, importances))
                
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
                    
            except Exception as e:
                logger.error(f"Error training model for WorkType {work_type}: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue with next work type instead of failing completely
                continue
        
        return models, feature_importances, metrics
        
    except Exception as e:
        logger.error(f"Error building models: {str(e)}")
        logger.error(traceback.format_exc())
        return {}, {}, {}


def validate_model_performance(model, X, y, work_type):
    """Check if model is learning properly"""
    
    # Check training performance
    train_pred = model.predict(X)
    train_mae = mean_absolute_error(y, train_pred)
    train_r2 = r2_score(y, train_pred)
    
    print(f"\n{work_type} Training Performance:")
    print(f"  Training MAE: {train_mae:.3f}")
    print(f"  Training R²: {train_r2:.3f}")
    print(f"  Target mean: {y.mean():.3f}")
    print(f"  Prediction mean: {train_pred.mean():.3f}")
    print(f"  Prediction/Target ratio: {train_pred.mean()/y.mean():.3f}")
    
    # Check if predictions are systematically biased
    bias = train_pred.mean() - y.mean()
    print(f"  Systematic bias: {bias:.3f}")
    
    return train_pred.mean() / y.mean()

def train_from_sql(connection_string=None, sql_query=None):
    """
    Train models using data from a SQL query with tiered feature system
    
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
        
            
        # Process data and train models using tiered feature system
        logger.info("Engineering features...")
        feature_df = engineer_features(df)

        logger.info("Creating lag features with tiered configuration...")
        
        # Use tiered configuration for lag features
        if FEATURE_TIERS['ADVANCED']:
            lag_days_to_use = LAG_DAYS
            rolling_windows_to_use = ROLLING_WINDOWS
        elif FEATURE_TIERS['INTERMEDIATE']:
            # Use intermediate configuration - defined in config
            lag_days_to_use = [1, 2, 3, 7, 14, 30]
            rolling_windows_to_use = [7, 14, 30]
        else:
            # Use basic configuration
            lag_days_to_use = [1, 7]
            rolling_windows_to_use = [7]
        
        logger.info(f"Using lag days: {lag_days_to_use}")
        logger.info(f"Using rolling windows: {rolling_windows_to_use}")
        
        lag_features_df = create_lag_features(
            feature_df,
            lag_days=lag_days_to_use,
            rolling_windows=rolling_windows_to_use
        )
        
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
        
        # Also save configuration info
        config_info = {
            'feature_tiers': FEATURE_TIERS,
            'active_tiers': [tier for tier, enabled in FEATURE_TIERS.items() if enabled],
            'training_date': datetime.now().isoformat(),
            'models_count': len(models)
        }
        
        config_file = os.path.join(MODELS_DIR, "training_config.json")
        import json
        with open(config_file, 'w') as f:
            json.dump(config_info, f, indent=2)
        
        logger.info(f"Model files saved successfully")
        logger.info(f"Performance summary saved to {performance_file}")
        logger.info(f"Training configuration saved to {config_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        logger.error(traceback.format_exc())
        return False
    
# In train_models.py, add this after loading data:
def diagnose_training_data(df, work_type):
    """Diagnose potential issues in training data"""
    wt_data = df[df['WorkType'] == work_type]
    
    print(f"\n=== Diagnosis for {work_type} ===")
    print(f"Total records: {len(wt_data)}")
    print(f"Date range: {wt_data['Date'].min()} to {wt_data['Date'].max()}")
    print(f"NoOfMan stats:")
    print(f"  Mean: {wt_data['NoOfMan'].mean():.2f}")
    print(f"  Median: {wt_data['NoOfMan'].median():.2f}")
    print(f"  Min: {wt_data['NoOfMan'].min():.2f}")
    print(f"  Max: {wt_data['NoOfMan'].max():.2f}")
    print(f"  Std: {wt_data['NoOfMan'].std():.2f}")
    
    # Check for outliers
    q1 = wt_data['NoOfMan'].quantile(0.25)
    q3 = wt_data['NoOfMan'].quantile(0.75)
    iqr = q3 - q1
    outliers = wt_data[(wt_data['NoOfMan'] < q1 - 1.5*iqr) | (wt_data['NoOfMan'] > q3 + 1.5*iqr)]
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(wt_data)*100:.1f}%)")
    
    # Check recent trend
    recent = wt_data.tail(30)
    print(f"Recent 30 days average: {recent['NoOfMan'].mean():.2f}")
    
    return wt_data['NoOfMan'].mean()


def main():
    """Main function to run the training process"""
    try:
        logger.info("Starting the model training process")
        
        # Log feature tier configuration
        active_tiers = [tier for tier, enabled in FEATURE_TIERS.items() if enabled]
        logger.info(f"Feature tiers enabled: {active_tiers}")
        
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

            # Use the feature engineering utilities with tiered configuration
            logger.info("Engineering features...")
            feature_df = engineer_features(df)
            
            logger.info("Creating lag features with tiered configuration...")
            
            # Use tiered configuration for lag features
            if FEATURE_TIERS['ADVANCED']:
                lag_days_to_use = LAG_DAYS
                rolling_windows_to_use = ROLLING_WINDOWS
            elif FEATURE_TIERS['INTERMEDIATE']:
                # Use intermediate configuration
                lag_days_to_use = [1, 2, 3, 7, 14, 30]
                rolling_windows_to_use = [7, 14, 30]
            else:
                # Use basic configuration
                lag_days_to_use = [1, 7]
                rolling_windows_to_use = [7]
            
            logger.info(f"Using lag days: {lag_days_to_use}")
            logger.info(f"Using rolling windows: {rolling_windows_to_use}")
            
            lag_features_df = create_lag_features(
                feature_df,
                lag_days=lag_days_to_use,
                rolling_windows=rolling_windows_to_use
            )
            
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