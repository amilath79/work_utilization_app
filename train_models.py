import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import MODELS_DIR
import openpyxl

# Ensure the models directory exists
# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
# Load and preprocess your data
def load_data(file_path):
    """Load and preprocess the work utilization data"""
    df = pd.read_excel(file_path)
    
    # Clean column names (remove whitespace)
    df.columns = df.columns.str.strip()
    
    # Ensure Date column is datetime 
    df['Date'] = pd.to_datetime(df['Date'])
    
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
    
    return df

# Feature engineering function
def engineer_features(df):
    """Create relevant features for the prediction model"""
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
    
    return data


def create_lag_features(data, group_col='WorkType', target_col='NoOfMan', lag_days=[1, 2, 3, 7, 14, 30]):
    """Create lag features for each WorkType's NoOfMan value"""
    # Make a copy of the input dataframe
    data_copy = data.copy()
    
    # First, check if there are any non-zero values in the target column
    non_zero_count = (data_copy[target_col] > 0).sum()
    print(f"Number of non-zero {target_col} values: {non_zero_count} out of {len(data_copy)}")
    
    # Ensure data is properly sorted by WorkType and Date (critical for time-series operations)  
    daily_data = data_copy.sort_values([group_col, 'Date']) 
    
    # Create lag features for each work type
    for lag in lag_days:
        daily_data[f'{target_col}_lag_{lag}'] = daily_data.groupby(group_col)[target_col].shift(lag)
    
    # Create rolling features by work type
    for work_type in daily_data[group_col].unique():
        mask = daily_data[group_col] == work_type
        work_type_data = daily_data.loc[mask, target_col]
        
        if len(work_type_data) >= 1:
            daily_data.loc[mask, f'{target_col}_rolling_mean_7'] = work_type_data.rolling(
                window=7, min_periods=1).mean()
            daily_data.loc[mask, f'{target_col}_rolling_max_7'] = work_type_data.rolling(
                window=7, min_periods=1).max()
            daily_data.loc[mask, f'{target_col}_rolling_min_7'] = work_type_data.rolling(
                window=7, min_periods=1).min()
            daily_data.loc[mask, f'{target_col}_rolling_std_7'] = work_type_data.rolling(
                window=7, min_periods=1).std().fillna(0)
    
    # Create same day of week lag
    daily_data[f'{target_col}_same_dow_lag'] = daily_data.groupby([group_col, 'DayOfWeek_feat'])[target_col].shift(1)
    
    # Fill NaN values with 0 instead of dropping
    daily_data = daily_data.fillna(0)
    return daily_data

# Build and train model for each WorkType
from sklearn.model_selection import TimeSeriesSplit

def build_models(processed_data, work_types, n_splits=5):
    """Build and train a model for each WorkType using time series cross-validation"""
    models = {}
    feature_importances = {}
    metrics = {}
    
    # Define features to use
    numeric_features = [
        'NoOfMan_lag_1', 'NoOfMan_lag_2', 'NoOfMan_lag_3', 'NoOfMan_lag_7', 
        'NoOfMan_lag_14', 'NoOfMan_lag_30', 'NoOfMan_rolling_mean_7',
        'NoOfMan_rolling_max_7', 'NoOfMan_rolling_min_7', 'NoOfMan_rolling_std_7',
        'NoOfMan_same_dow_lag', 'IsWeekend_feat'
    ]
    
    categorical_features = ['DayOfWeek_feat', 'Month_feat']
    
    # Function to calculate modified MAPE with minimum threshold
    def modified_mape(y_true, y_pred, epsilon=1.0):
        """Calculate MAPE with a minimum threshold to avoid division by zero"""
        denominator = np.maximum(np.abs(y_true), epsilon)
        return np.mean(np.abs(y_pred - y_true) / denominator) * 100
    
    for work_type in work_types:
        print(f"Building model for WorkType: {work_type}")
        
        # Filter data for this WorkType
        work_type_data = processed_data[processed_data['WorkType'] == work_type] 
        
        if len(work_type_data) < 30:  # Skip if not enough data
            print(f"  Skipping {work_type}: Not enough data ({len(work_type_data)} records)")
            continue
        
        # Sort data by date to ensure time-based splitting works correctly
        work_type_data = work_type_data.sort_values('Date')
        
        # Prepare features and target
        X = work_type_data[numeric_features + categorical_features]
        y = work_type_data['NoOfMan']
        
        # Define preprocessing for categorical features  
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Define the model pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
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
        all_feature_names = cat_feature_names + numeric_features
        
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
        
        print(f"  Model for {work_type} - MAE: {metrics[work_type]['MAE']:.4f}, RMSE: {metrics[work_type]['RMSE']:.4f}, R²: {metrics[work_type]['R²']:.4f}, MAPE: {metrics[work_type]['MAPE']:.2f}%")
        
        # Also print individual fold scores for detailed analysis
        print(f"  Cross-validation details:")
        for i in range(len(mae_scores)):
            print(f"    Fold {i+1}: MAE={mae_scores[i]:.4f}, RMSE={rmse_scores[i]:.4f}, R²={r2_scores[i]:.4f}, MAPE={mape_scores[i]:.2f}%")
    
    return models, feature_importances, metrics

if __name__ == "__main__":
    file_path = "C:/forlogssystems/Data/work_utilization_melted1.xlsx"
    df = load_data(file_path)
    
    feature_df = engineer_features(df)
    lag_features_df = create_lag_features(feature_df)
    
    work_types = lag_features_df['WorkType'].unique()
    models, feature_importances, metrics = build_models(lag_features_df, work_types, n_splits=5)
    
    # Create the Models directory if it doesn't exist
    models_dir = "C:/forlogssystems/Models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Saving model files...")
    with open(os.path.join(models_dir, "work_utilization_models.pkl"), "wb") as f:
        pickle.dump(models, f)
        
    with open(os.path.join(models_dir, "work_utilization_feature_importances.pkl"), "wb") as f:
        pickle.dump(feature_importances, f)
        
    with open(os.path.join(models_dir, "work_utilization_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
        
    print(f"Successfully saved {len(models)} models to {models_dir}!")