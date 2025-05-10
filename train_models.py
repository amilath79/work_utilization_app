"""
Module for training workforce prediction models with productivity metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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

def engineer_features(df):
    """
    Create relevant features for the prediction model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    logger.info("Engineering features")
    
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Extract date features
    data['Year_feat'] = data['Date'].dt.year
    data['Month_feat'] = data['Date'].dt.month  
    data['DayOfMonth'] = data['Date'].dt.day
    data['DayOfWeek_feat'] = data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    data['Quarter'] = data['Date'].dt.quarter
    data['IsWeekend_feat'] = data['DayOfWeek_feat'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add week of year
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week
    
    # Add day of year
    data['DayOfYear'] = data['Date'].dt.dayofyear
    
    # Calculate days since the start of the dataset
    min_date = data['Date'].min() 
    data['DaysSinceStart'] = (data['Date'] - min_date).dt.days
    
    # Process productivity metrics if they exist
    has_productivity_metrics = all(col in data.columns for col in ['Hours', 'SystemHours', 'Quantity'])
    
    if has_productivity_metrics:
        logger.info("Found productivity metrics, creating derived features")
        
        # Create ratio between Hours and SystemHours
        data['Hours_SystemHours_Ratio'] = np.where(
            data['SystemHours'] > 0,
            data['Hours'] / data['SystemHours'],
            1
        )
        
        # Create productivity per worker metrics
        data['Quantity_per_Worker'] = np.where(
            data['NoOfMan'] > 0,
            data['Quantity'] / data['NoOfMan'],
            0
        )
        
        # Combined KPI metric (balanced between Resource and System)
        if all(col in data.columns for col in ['ResourceKPI', 'SystemKPI']):
            data['Combined_KPI'] = (data['ResourceKPI'] + data['SystemKPI']) / 2
        
        # Relative workload compared to typical for this work type
        data['Relative_Quantity'] = data['Quantity'] / data.groupby('WorkType')['Quantity'].transform('mean').replace(0, 1)
        
        # Hours per quantity (complexity indicator)
        data['Hours_per_Quantity'] = np.where(
            data['Quantity'] > 0,
            data['Hours'] / data['Quantity'],
            0
        )
        
        # System hours per quantity
        data['SystemHours_per_Quantity'] = np.where(
            data['Quantity'] > 0,
            data['SystemHours'] / data['Quantity'],
            0
        )
        
        logger.info(f"Created {len(data.columns) - len(df.columns)} new features")
    else:
        logger.info("No productivity metrics found in the dataset")
    
    return data

def create_lag_features(data, group_col='WorkType', target_col='NoOfMan', lag_days=None, rolling_windows=None):
    """
    Create lag features for each WorkType's NoOfMan value and productivity metrics
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame with engineered features
    group_col : str
        Column to group by (default: 'WorkType')
    target_col : str
        Target column to create lag features for (default: 'NoOfMan')
    lag_days : list
        List of lag days to create
    rolling_windows : list
        List of rolling window sizes to create
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features
    """
    # Use config parameters if not specified
    if lag_days is None:
        lag_days = LAG_DAYS
    
    if rolling_windows is None:
        rolling_windows = ROLLING_WINDOWS
    
    logger.info(f"Creating lag features with lag days: {lag_days}")
    logger.info(f"Creating rolling window features with windows: {rolling_windows}")
    
    # Make a copy of the input dataframe
    data_copy = data.copy()
    
    # First, check if there are any non-zero values in the target column
    non_zero_count = (data_copy[target_col] > 0).sum()
    logger.info(f"Number of non-zero {target_col} values: {non_zero_count} out of {len(data_copy)}")
    
    # Ensure data is properly sorted by WorkType and Date (critical for time-series operations)  
    daily_data = data_copy.sort_values([group_col, 'Date'])
    
    # Create lag features for the target column (NoOfMan)
    for lag in lag_days:
        daily_data[f'{target_col}_lag_{lag}'] = daily_data.groupby(group_col)[target_col].shift(lag)
    
    # Create rolling features for the target column
    for window in rolling_windows:
        # Rolling mean for all windows
        daily_data[f'{target_col}_rolling_mean_{window}'] = daily_data.groupby(group_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        
        # Additional statistics just for 7-day window (to avoid feature explosion)
        if window == 7:
            daily_data[f'{target_col}_rolling_max_{window}'] = daily_data.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max())
            daily_data[f'{target_col}_rolling_min_{window}'] = daily_data.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min())
            daily_data[f'{target_col}_rolling_std_{window}'] = daily_data.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
    
    # Create same day of week lag for NoOfMan
    daily_data[f'{target_col}_same_dow_lag'] = daily_data.groupby([group_col, 'DayOfWeek_feat'])[target_col].shift(1)
    
    # Create same day of month lag
    daily_data[f'{target_col}_same_dom_lag'] = daily_data.groupby([group_col, 'DayOfMonth'])[target_col].shift(1)
    
    # Create lag features for productivity metrics if they exist
    productivity_metrics = [
        'Quantity', 'ResourceKPI', 'SystemKPI', 'Combined_KPI',
        'Hours_SystemHours_Ratio', 'Quantity_per_Worker', 'Relative_Quantity',
        'Hours_per_Quantity', 'SystemHours_per_Quantity'
    ]
    
    # Check which productivity metrics exist in the data
    available_metrics = [metric for metric in productivity_metrics if metric in daily_data.columns]
    
    if available_metrics:
        logger.info(f"Creating lag features for productivity metrics: {available_metrics}")
        
        for metric in available_metrics:
            # Create 1-day and 7-day lags for main productivity metrics
            for lag in [1, 7]:
                if lag in lag_days:
                    daily_data[f'{metric}_lag_{lag}'] = daily_data.groupby(group_col)[metric].shift(lag)
            
            # Create 7-day rolling mean for productivity metrics
            daily_data[f'{metric}_rolling_mean_7'] = daily_data.groupby(group_col)[metric].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean())
    
    # Create quantity-based same-day-of-week lag if available
    if 'Quantity' in daily_data.columns:
        daily_data['Quantity_same_dow_lag'] = daily_data.groupby([group_col, 'DayOfWeek_feat'])['Quantity'].shift(1)
    
    # Create trend features if both lag 1 and lag 7 are available
    if 1 in lag_days and 7 in lag_days:
        # NoOfMan trends
        daily_data[f'{target_col}_7day_trend'] = daily_data[target_col] - daily_data[f'{target_col}_lag_7']
        daily_data[f'{target_col}_1day_trend'] = daily_data[target_col] - daily_data[f'{target_col}_lag_1']
        
        # Quantity trends if available
        if 'Quantity' in daily_data.columns and 'Quantity_lag_1' in daily_data.columns and 'Quantity_lag_7' in daily_data.columns:
            daily_data['Quantity_7day_trend'] = daily_data['Quantity'] - daily_data['Quantity_lag_7']
            daily_data['Quantity_1day_trend'] = daily_data['Quantity'] - daily_data['Quantity_lag_1']
    
    # Create predictive features if quantity data is available
    if 'Quantity' in daily_data.columns and 'Quantity_lag_1' in daily_data.columns:
        # Calculate average NoOfMan per Quantity for each work type
        avg_workers_per_unit = daily_data.groupby(group_col).apply(
            lambda x: (x[target_col] / x['Quantity']).replace([np.inf, -np.inf], np.nan).mean()
        ).fillna(0)
        
        # Map this average back to each row
        daily_data['avg_workers_per_unit'] = daily_data[group_col].map(avg_workers_per_unit)
        
        # Predict workers based on previous day's quantity
        daily_data['Workers_Predicted_from_Quantity'] = daily_data['Quantity_lag_1'] * daily_data['avg_workers_per_unit']
    
    # Fill NaN values with 0 instead of dropping
    daily_data = daily_data.fillna(0)
    
    logger.info(f"Final shape after creating lag features: {daily_data.shape}")
    return daily_data

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
    # If work_types is not provided, get them from the data
    if work_types is None:
        work_types = sorted(processed_data['WorkType'].unique())
    
    logger.info(f"Building models for {len(work_types)} work types")
    
    models = {}
    feature_importances = {}
    metrics = {}
    
    # Define base features to use - USING CONFIG VALUES
    base_numeric_features = ['IsWeekend_feat']
    
    # Add all lag features from LAG_DAYS config
    for lag in LAG_DAYS:
        base_numeric_features.append(f'NoOfMan_lag_{lag}')
    
    # Add all rolling window features from ROLLING_WINDOWS config
    for window in ROLLING_WINDOWS:
        base_numeric_features.append(f'NoOfMan_rolling_mean_{window}')
        if window == 7:  # Only include max/min/std for 7-day window
            base_numeric_features.append(f'NoOfMan_rolling_max_{window}')
            base_numeric_features.append(f'NoOfMan_rolling_min_{window}')
            base_numeric_features.append(f'NoOfMan_rolling_std_{window}')
    
    # Add same day of week/month lag
    base_numeric_features.extend(['NoOfMan_same_dow_lag', 'NoOfMan_same_dom_lag'])
    
    # Add trend features if configured
    if 1 in LAG_DAYS and 7 in LAG_DAYS:
        base_numeric_features.extend(['NoOfMan_1day_trend', 'NoOfMan_7day_trend'])
    
    # Define productivity features
    productivity_features = [
        # Lag features for main metrics
        'Quantity_lag_1', 'Quantity_lag_7', 
        'ResourceKPI_lag_1', 'SystemKPI_lag_1', 
        'Combined_KPI_lag_1',
        
        # Rolling means
        'Quantity_rolling_mean_7',
        'ResourceKPI_rolling_mean_7',
        'SystemKPI_rolling_mean_7',
        
        # Derived metrics
        'Hours_SystemHours_Ratio',
        'Quantity_per_Worker',
        'Relative_Quantity',
        'Quantity_same_dow_lag',
        'Hours_per_Quantity',
        'SystemHours_per_Quantity',
        
        # Trend features
        'Quantity_7day_trend',
        'Quantity_1day_trend',
        
        # Predictive features
        'Workers_Predicted_from_Quantity'
    ]
    
    # Categorical features for one-hot encoding
    categorical_features = ['DayOfWeek_feat', 'Month_feat']
    
    # Function to calculate modified MAPE with minimum threshold
    def modified_mape(y_true, y_pred, epsilon=1.0):
        """Calculate MAPE with a minimum threshold to avoid division by zero"""
        denominator = np.maximum(np.abs(y_true), epsilon)
        return np.mean(np.abs(y_pred - y_true) / denominator) * 100
    
    # Log all features that will be used
    logger.info(f"Base numeric features: {base_numeric_features}")
    logger.info(f"Productivity features (if available): {productivity_features}")
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
        
        # Check which features are available in the dataset
        available_base_numeric = [f for f in base_numeric_features if f in work_type_data.columns]
        available_productivity = [f for f in productivity_features if f in work_type_data.columns]
        all_numeric_features = available_base_numeric + available_productivity
        
        logger.info(f"Using {len(all_numeric_features)} numeric features and {len(categorical_features)} categorical features")
        
        # Prepare features and target
        X = work_type_data[all_numeric_features + categorical_features]
        y = work_type_data['NoOfMan']
        
        # Define preprocessing for categorical features  
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Define the model pipeline using DEFAULT_MODEL_PARAMS from config
        model_params = DEFAULT_MODEL_PARAMS.copy()
        model_params["n_estimators"] = 200  # Override to use more trees
        model_params["max_depth"] = 15      # Deeper trees for more complex relationships
        
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
        ohe = pipeline.named_steps['preprocessor'].transformers_[0][1]
        
        # Extract feature names after one-hot encoding
        cat_feature_names = []
        for i, feature in enumerate(categorical_features):
            categories = ohe.categories_[i]
            for category in categories:
                cat_feature_names.append(f"{feature}_{category}")
        
        # Combine with numeric feature names
        all_feature_names = cat_feature_names + all_numeric_features
        
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
            SELECT Date, PunchCode, Hours, NoOfMan, SystemHours, Quantity, ResourceKPI, SystemKPI 
            FROM WorkUtilizationData
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
        
        # Process data and train models
        feature_df = engineer_features(df)
        lag_features_df = create_lag_features(feature_df)
        
        work_types = lag_features_df['WorkType'].unique()
        logger.info(f"Found {len(work_types)} unique work types")
        
        models, feature_importances, metrics = build_models(lag_features_df, work_types, n_splits=5)
        
        # Save models
        save_models(models, feature_importances, metrics)
        
        return models, feature_importances, metrics
        
    except Exception as e:
        logger.error(f"Error in train_from_sql: {str(e)}")
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
        
        performance_df = pd.DataFrame(performance_summary).sort_values('R²', ascending=False)
        performance_file = os.path.join(MODELS_DIR, "model_performance_summary.xlsx")
        performance_df.to_excel(performance_file, index=False)
        
        logger.info(f"Model files saved successfully")
        logger.info(f"Performance summary saved to {performance_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving models: {str(e)}")
        return False

def main():
    """Main function to run the training process"""
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
        train_from_sql()
    else:
        # Train using Excel file
        file_path = os.path.join(DATA_DIR, "work_utilization_melted1.csv")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            # Ask user for file path
            file_path = input("Enter path to data file (.xlsx or .csv): ")
        
        df = load_data(file_path)
        feature_df = engineer_features(df)
        lag_features_df = create_lag_features(feature_df)
        
        work_types = lag_features_df['WorkType'].unique()
        logger.info(f"Found {len(work_types)} unique work types")
        
        models, feature_importances, metrics = build_models(lag_features_df, work_types, n_splits=5)
        save_models(models, feature_importances, metrics)
    
    logger.info("Model training completed")

if __name__ == "__main__":
    main()