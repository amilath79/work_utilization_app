import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, Easter, TH, FR, SA

# Define the Swedish Holiday Calendar class
class SwedishHolidayCalendar(AbstractHolidayCalendar):
    """
    Swedish Holiday Calendar
    """
    rules = [
        Holiday('New Year\'s Day', month=1, day=1),
        Holiday('Epiphany', month=1, day=6),
        Holiday('Good Friday', month=3, day=23, offset=[pd.DateOffset(days=-2), Easter()]),
        Holiday('Easter Sunday', month=3, day=23, offset=[Easter()]),
        Holiday('Easter Monday', month=3, day=23, offset=[pd.DateOffset(days=1), Easter()]),
        Holiday('Labour Day', month=5, day=1),
        Holiday('Ascension Day', month=5, day=1, offset=[pd.DateOffset(days=39), Easter()]),
        Holiday('National Day', month=6, day=6),
        Holiday('Midsummer\'s Eve', month=6, day=19, offset=pd.DateOffset(weekday=TH(3))),
        Holiday('Midsummer\'s Day', month=6, day=20, offset=pd.DateOffset(weekday=FR(4))),
        Holiday('All Saints\' Day', month=10, day=31, offset=pd.DateOffset(weekday=SA(5))),
        Holiday('Christmas Eve', month=12, day=24),
        Holiday('Christmas Day', month=12, day=25),
        Holiday('Second Day of Christmas', month=12, day=26),
        Holiday('New Year\'s Eve', month=12, day=31)
    ]

# Function to calculate Symmetric Mean Absolute Percentage Error (SMAPE)
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100 if np.sum(denominator) > 0 else 0

# Function to calculate Mean Absolute Scaled Error (MASE)
def mase(y_true, y_pred, y_train):
    n = len(y_true)
    d = np.sum(np.abs(y_train[1:] - y_train[:-1])) / (len(y_train) - 1)
    errors = np.abs(y_true - y_pred)
    return np.mean(errors) / d if d > 0 else (0 if np.mean(errors) == 0 else float('inf'))


# --- Step 1: Load Data ---
file_path = 'enhanced_df.xlsx'
df = pd.read_excel(file_path)

# --- Step 2: Filter Data ---
df_209 = df[df['WorkType'] == 209].copy()

# --- Step 3: Handle Dates and Holidays ---
df_209['Date'] = pd.to_datetime(df_209['Date'])
cal = SwedishHolidayCalendar()
swedish_holidays = cal.holidays(start=df_209['Date'].min(), end=df_209['Date'].max())
df_209_working_days = df_209[~((df_209['Date'].dt.dayofweek >= 5) | (df_209['Date'].isin(swedish_holidays)))].copy()

if 'Date' in df_209_working_days.columns:
    df_209_working_days = df_209_working_days.set_index('Date')

# Calculate the ResourceKPI
df_209_working_days['ResourceKPI'] = df_209_working_days['Quantity'] / df_209_working_days['Hours']

# Create date-related features
df_209_working_days['year'] = df_209_working_days.index.year
df_209_working_days['month'] = df_209_working_days.index.month
df_209_working_days['week_no'] = df_209_working_days.index.isocalendar().week.astype(int)
df_209_working_days['day_of_week'] = df_209_working_days.index.dayofweek
df_209_working_days['is_weekend'] = (df_209_working_days.index.dayofweek >= 5).astype(int)
df_209_working_days['quarter'] = df_209_working_days.index.quarter
df_209_working_days['is_monthend'] = df_209_working_days.index.is_month_end.astype(int)
df_209_working_days['is_monthstart'] = df_209_working_days.index.is_month_start.astype(int)

# Create the UtilizationRatio feature
epsilon = 1e-9
df_209_working_days['UtilizationRatio'] = df_209_working_days['SystemHours'] / (df_209_working_days['Hours'] + epsilon)
df_209_working_days['UtilizationRatio'] = df_209_working_days['UtilizationRatio'].replace([float('inf'), float('-inf')], 0)

# Define lags and windows
LAGS = [1, 7, 14, 21, 30]
WINDOWS = [7, 14]
COLUMNS_TO_LAG_WINDOW = ['Quantity', 'SystemHours', 'Hours']

# Create lagged features
for col in COLUMNS_TO_LAG_WINDOW:
    for lag in LAGS:
        df_209_working_days[f'{col}_Lag{lag}'] = df_209_working_days[col].shift(lag)

# Create rolling window features (mean)
for col in COLUMNS_TO_LAG_WINDOW:
    for window in WINDOWS:
        df_209_working_days[f'{col}_Window{window}_Mean'] = df_209_working_days[col].rolling(window=window, min_periods=1).mean()
        df_209_working_days[f'{col}_Window{window}_Std'] = df_209_working_days[col].rolling(window=window, min_periods=1).std()
        df_209_working_days[f'{col}_Window{window}_Max'] = df_209_working_days[col].rolling(window=window, min_periods=1).max()
        df_209_working_days[f'{col}_Window{window}_Min'] = df_209_working_days[col].rolling(window=window, min_periods=1).min()
        df_209_working_days[f'{col}_Window{window}_Median'] = df_209_working_days[col].rolling(window=window, min_periods=1).median()
        df_209_working_days[f'{col}_Window{window}_Sum'] = df_209_working_days[col].rolling(window=window, min_periods=1).sum()
        df_209_working_days[f'{col}_Window{window}_EWMA'] = df_209_working_days[col].ewm(span=window, min_periods=1).mean()
        df_209_working_days[f'{col}_Window{window}_CV'] = df_209_working_days[col].rolling(window=window, min_periods=1).std() / df_209_working_days[col].rolling(window=window, min_periods=1).mean()
        df_209_working_days[f'{col}_Window{window}_IQR'] = df_209_working_days[col].rolling(window=window, min_periods=1).quantile(0.75) - df_209_working_days[col].rolling(window=window, min_periods=1).quantile(0.25)


# Create last year same week same day features
def create_last_year_features(df):
    """Create features for last year same week same day"""
    df = df.copy()
    
    # Sort by date to ensure proper alignment
    df = df.sort_index()
    
    # Initialize the new columns
    df['Quantity_LastYear'] = np.nan
    df['Hours_LastYear'] = np.nan
    df['SystemHours_LastYear'] = np.nan
    
    for current_date in df.index:
        # Find the same week and same day of week from last year
        last_year_date = current_date - pd.DateOffset(years=1)
        
        # Create a window of ±3 days around the target date to find the closest match
        start_window = last_year_date - pd.DateOffset(days=3)
        end_window = last_year_date + pd.DateOffset(days=3)
        
        # Filter data within the window
        window_data = df[(df.index >= start_window) & (df.index <= end_window)]
        
        if not window_data.empty:
            # Find the date with the same day of week, or closest match
            current_dow = current_date.dayofweek
            window_data_with_dow = window_data.copy()
            window_data_with_dow['dow_diff'] = abs(window_data_with_dow.index.dayofweek - current_dow)
            
            # Sort by day of week difference and then by date difference
            window_data_with_dow['date_diff'] = abs((window_data_with_dow.index - last_year_date).days)
            best_match = window_data_with_dow.sort_values(['dow_diff', 'date_diff']).iloc[0]
            
            # Assign the values
            df.loc[current_date, 'Quantity_LastYear'] = best_match['Quantity']
            df.loc[current_date, 'Hours_LastYear'] = best_match['Hours']
            df.loc[current_date, 'SystemHours_LastYear'] = best_match['SystemHours']
    
    return df

# Apply the function to create last year features
df_209_working_days = create_last_year_features(df_209_working_days)

# Create interaction features
df_209_working_days['ResourceKPI_Quantity_Lag7'] = df_209_working_days['ResourceKPI'] * df_209_working_days['Quantity_Lag7']
df_209_working_days['ResourceKPI_Quantity_Lag14'] = df_209_working_days['ResourceKPI'] * df_209_working_days['Quantity_Lag14']

# Create interaction features with last year data
df_209_working_days['ResourceKPI_Quantity_LastYear'] = df_209_working_days['ResourceKPI'] * df_209_working_days['Quantity_LastYear']

# Handle remaining NaN values introduced by lagging and rolling window calculations
df_209_working_days = df_209_working_days.ffill().bfill()

# Clean feature names to remove special characters that cause LightGBM issues
def clean_feature_names(df):
    """Clean column names to be compatible with LightGBM"""
    df = df.copy()
    new_columns = {}
    for col in df.columns:
        # Replace special characters that cause issues in LightGBM
        clean_col = str(col)
        # Remove or replace all problematic characters
        clean_col = clean_col.replace("'", "")
        clean_col = clean_col.replace('"', '')
        clean_col = clean_col.replace('[', '')
        clean_col = clean_col.replace(']', '')
        clean_col = clean_col.replace('<', '')
        clean_col = clean_col.replace('>', '')
        clean_col = clean_col.replace(',', '_')
        clean_col = clean_col.replace(' ', '_')
        clean_col = clean_col.replace('(', '')
        clean_col = clean_col.replace(')', '')
        clean_col = clean_col.replace(':', '_')
        clean_col = clean_col.replace(';', '_')
        clean_col = clean_col.replace('{', '')
        clean_col = clean_col.replace('}', '')
        clean_col = clean_col.replace('\\', '_')
        clean_col = clean_col.replace('/', '_')
        clean_col = clean_col.replace('|', '_')
        clean_col = clean_col.replace('&', 'and')
        clean_col = clean_col.replace('%', 'pct')
        clean_col = clean_col.replace('#', 'num')
        clean_col = clean_col.replace('@', 'at')
        clean_col = clean_col.replace('!', '')
        clean_col = clean_col.replace('?', '')
        clean_col = clean_col.replace('*', '')
        clean_col = clean_col.replace('+', 'plus')
        clean_col = clean_col.replace('-', '_')
        clean_col = clean_col.replace('=', 'eq')
        clean_col = clean_col.replace('

# --- Step 7: Prepare Data for Modeling ---
exclude_columns = ['Hours', 'Date', 'WorkType', 'Quantity', 'SystemHours']

# Dynamically generate features excluding unnecessary columns and target
all_columns = df_209_working_days.columns.tolist()
features = [col for col in all_columns if col not in exclude_columns]
target = 'Hours'

# Remove duplicate feature names (if any, although the list should be unique now)
features = list(dict.fromkeys(features))

# Additional check: ensure target column exists
if target not in df_209_working_days.columns:
    raise ValueError(f"Target column '{target}' not found in dataframe columns: {df_209_working_days.columns.tolist()}")

X = df_209_working_days[features]
y = df_209_working_days[target]

print(f"Features selected: {len(features)}")
print(f"Target variable: {target}")
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Check for any remaining NaN values
if X.isnull().any().any():
    print("Warning: NaN values found in features")
    print("NaN counts by column:")
    nan_counts = X.isnull().sum()
    print(nan_counts[nan_counts > 0])
    # Fill remaining NaNs with 0
    X = X.fillna(0)

if y.isnull().any():
    print("Warning: NaN values found in target variable")
    print(f"Number of NaN values in target: {y.isnull().sum()}")
    # Remove rows where target is NaN
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    print(f"After removing NaN targets: X={X.shape}, y={y.shape}")

# --- Step 8 & 9: Time Series Cross-Validation Split and Evaluation ---
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Store evaluation metrics
cv_scores = {
    'R2': [],
    'MSE': [],
    'RMSE': [],
    'MAE': [],
    'MASE': [],
    'SMAPE': []
}

print("Cross-Validation Results:")
print("-" * 50)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    model = lgb.LGBMRegressor(
        learning_rate=0.01, 
        min_child_samples=20, 
        n_estimators=300, 
        num_leaves=31, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    mase_score = mase(y_val, y_pred, y_train)
    smape_score = smape(y_val, y_pred)
    
    # Store scores
    cv_scores['R2'].append(r2)
    cv_scores['MSE'].append(mse)
    cv_scores['RMSE'].append(rmse)
    cv_scores['MAE'].append(mae)
    cv_scores['MASE'].append(mase_score)
    cv_scores['SMAPE'].append(smape_score)
    
    print(f"Fold {fold + 1}:")
    print(f"  R²: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MASE: {mase_score:.4f}")
    print(f"  SMAPE: {smape_score:.2f}%")
    print()

# Calculate average scores
print("Average Cross-Validation Scores:")
print("=" * 50)
for metric, scores in cv_scores.items():
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    if metric == 'SMAPE':
        print(f"{metric}: {avg_score:.2f}% ± {std_score:.2f}%")
    else:
        print(f"{metric}: {avg_score:.4f} ± {std_score:.4f}")

# --- Train Final Model on Full Dataset ---
print("\n" + "=" * 50)
print("Training Final Model on Full Dataset")
print("=" * 50)

best_params = {'learning_rate': 0.01, 'min_child_samples': 20, 'n_estimators': 300, 'num_leaves': 31}
best_lgbm_model = lgb.LGBMRegressor(**best_params, random_state=42)
best_lgbm_model.fit(X, y)

# --- Predict Hours for Next 7 Working Days ---
last_date_in_data = df_209_working_days.index.max()

def find_next_working_days(start_date, n, holiday_calendar):
    next_days = []
    current_date = start_date + timedelta(days=1)
    future_holidays = holiday_calendar.holidays(start=current_date - timedelta(days=365), end=current_date + timedelta(days=365 * 2))
    while len(next_days) < n:
        if current_date.dayofweek < 5 and current_date not in future_holidays:
            next_days.append(current_date)
        current_date += timedelta(days=1)
    return next_days

def get_last_year_values(df, target_date, columns=['Quantity', 'Hours', 'SystemHours']):
    """Get last year same week same day values for prediction"""
    last_year_date = target_date - pd.DateOffset(years=1)
    
    # Create a window of ±7 days around the target date to find working days
    start_window = last_year_date - pd.DateOffset(days=7)
    end_window = last_year_date + pd.DateOffset(days=7)
    
    # Filter data within the window (only working days)
    window_data = df[(df.index >= start_window) & (df.index <= end_window)]
    
    if window_data.empty:
        return {f'{col}_LastYear': np.nan for col in columns}
    
    # Find the date with the same day of week, or closest match
    target_dow = target_date.dayofweek
    window_data_with_dow = window_data.copy()
    window_data_with_dow['dow_diff'] = abs(window_data_with_dow.index.dayofweek - target_dow)
    
    # Sort by day of week difference and then by date difference
    window_data_with_dow['date_diff'] = abs((window_data_with_dow.index - last_year_date).days)
    best_match = window_data_with_dow.sort_values(['dow_diff', 'date_diff']).iloc[0]
    
    # Return the values
    return {f'{col}_LastYear': best_match[col] for col in columns}

cal = SwedishHolidayCalendar()
next_7_working_days = find_next_working_days(last_date_in_data, 7, cal)

future_features_list = []
last_row_data = df_209_working_days.iloc[-1]
future_kpi_value = 293

for pred_date in next_7_working_days:
    future_row = {}
    future_row['month'] = pred_date.month
    future_row['year'] = pred_date.year
    future_row['week_no'] = pred_date.isocalendar().week
    future_row['day_of_week'] = pred_date.dayofweek
    future_row['is_weekend'] = int(pred_date.dayofweek >= 5)
    future_row['quarter'] = pred_date.quarter
    future_row['is_monthend'] = int(pd.to_datetime(pred_date).is_month_end)
    future_row['is_monthstart'] = int(pd.to_datetime(pred_date).is_month_start)
    future_row['ResourceKPI'] = future_kpi_value
    future_row['UtilizationRatio'] = last_row_data['UtilizationRatio']

    # Define lags and windows (re-defined for clarity)
    LAGS = [1, 7, 14, 21, 30]
    WINDOWS = [7, 14]
    COLUMNS_TO_LAG_WINDOW = ['Quantity', 'SystemHours', 'Hours']

    for col in COLUMNS_TO_LAG_WINDOW:
        for lag in LAGS:
            lag_col_name = f'{col}_Lag{lag}'
            future_row[lag_col_name] = last_row_data.get(lag_col_name, np.nan)

        for window in WINDOWS:
            for stat in ['Mean', 'Std', 'Max', 'Min', 'Median', 'Sum', 'EWMA', 'CV', 'IQR']:
                window_col_name = f'{col}_Window{window}_{stat}'
                future_row[window_col_name] = last_row_data.get(window_col_name, np.nan)

    # Get last year same week same day values
    last_year_values = get_last_year_values(df_209_working_days, pred_date)
    future_row.update(last_year_values)

    # Interaction features
    qty_lag7 = future_row.get('Quantity_Lag7', np.nan)
    qty_lag14 = future_row.get('Quantity_Lag14', np.nan)
    qty_last_year = future_row.get('Quantity_LastYear', np.nan)
    
    future_row['ResourceKPI_Quantity_Lag7'] = future_kpi_value * qty_lag7
    future_row['ResourceKPI_Quantity_Lag14'] = future_kpi_value * qty_lag14
    future_row['ResourceKPI_Quantity_LastYear'] = future_kpi_value * qty_last_year

    future_features_list.append(future_row)

X_future = pd.DataFrame(future_features_list, index=next_7_working_days)

# Ensure feature order matches training data (X) and handle potential missing columns in X_future
X_future = X_future.reindex(columns=X.columns, fill_value=np.nan)

predicted_hours = best_lgbm_model.predict(X_future)
predictions = pd.DataFrame({'Predicted_Hours': predicted_hours}, index=next_7_working_days)

print("\nPredicted Hours for the next 7 valid working days:")
print("=" * 50)
for date, pred_hours in predictions.iterrows():
    print(f"{date.strftime('%Y-%m-%d (%A)')}: {pred_hours['Predicted_Hours']:.2f} hours")

print(f"\nTotal predicted hours for next 7 working days: {predicted_hours.sum():.2f} hours")
print(f"Average predicted hours per day: {predicted_hours.mean():.2f} hours"), 'dollar')
        clean_col = clean_col.replace('^', '')
        clean_col = clean_col.replace('~', '')
        clean_col = clean_col.replace('`', '')
        # Remove any remaining non-alphanumeric characters except underscores
        import re
        clean_col = re.sub(r'[^a-zA-Z0-9_]', '_', clean_col)
        # Remove multiple consecutive underscores
        clean_col = re.sub(r'_+', '_', clean_col)
        # Remove leading/trailing underscores
        clean_col = clean_col.strip('_')
        # Ensure it starts with a letter or underscore
        if clean_col and not (clean_col[0].isalpha() or clean_col[0] == '_'):
            clean_col = 'col_' + clean_col
        new_columns[col] = clean_col
    
    # Check for duplicates and make them unique
    seen = {}
    final_columns = {}
    for old_col, new_col in new_columns.items():
        if new_col in seen:
            counter = 1
            original_new_col = new_col
            while new_col in seen:
                new_col = f"{original_new_col}_{counter}"
                counter += 1
        seen[new_col] = True
        final_columns[old_col] = new_col
    
    df = df.rename(columns=final_columns)
    return df

df_209_working_days = clean_feature_names(df_209_working_days)

# Debug: Print feature names to check for any remaining issues
print("Checking feature names for special characters:")
problematic_features = []
for col in df_209_working_days.columns:
    # Check for any characters that might cause JSON issues
    if any(char in str(col) for char in ['"', "'", '[', ']', '{', '}', '\\', '\n', '\r', '\t']):
        problematic_features.append(col)

if problematic_features:
    print(f"Found problematic features: {problematic_features}")
else:
    print("All feature names appear clean.")

print(f"Total number of features: {len(df_209_working_days.columns)}")

# --- Step 7: Prepare Data for Modeling ---
exclude_columns = ['Hours', 'Date', 'WorkType', 'Quantity', 'SystemHours']

# Dynamically generate features excluding unnecessary columns and target
all_columns = df_209_working_days.columns.tolist()
features = [col for col in all_columns if col not in exclude_columns]
target = 'Hours'

# Remove duplicate feature names (if any, although the list should be unique now)
features = list(dict.fromkeys(features))

X = df_209_working_days[features]
y = df_209_working_days[target]

# --- Step 8 & 9: Time Series Cross-Validation Split and Evaluation ---
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Store evaluation metrics
cv_scores = {
    'R2': [],
    'MSE': [],
    'RMSE': [],
    'MAE': [],
    'MASE': [],
    'SMAPE': []
}

print("Cross-Validation Results:")
print("-" * 50)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    model = lgb.LGBMRegressor(
        learning_rate=0.01, 
        min_child_samples=20, 
        n_estimators=300, 
        num_leaves=31, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    mase_score = mase(y_val, y_pred, y_train)
    smape_score = smape(y_val, y_pred)
    
    # Store scores
    cv_scores['R2'].append(r2)
    cv_scores['MSE'].append(mse)
    cv_scores['RMSE'].append(rmse)
    cv_scores['MAE'].append(mae)
    cv_scores['MASE'].append(mase_score)
    cv_scores['SMAPE'].append(smape_score)
    
    print(f"Fold {fold + 1}:")
    print(f"  R²: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MASE: {mase_score:.4f}")
    print(f"  SMAPE: {smape_score:.2f}%")
    print()

# Calculate average scores
print("Average Cross-Validation Scores:")
print("=" * 50)
for metric, scores in cv_scores.items():
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    if metric == 'SMAPE':
        print(f"{metric}: {avg_score:.2f}% ± {std_score:.2f}%")
    else:
        print(f"{metric}: {avg_score:.4f} ± {std_score:.4f}")

# --- Train Final Model on Full Dataset ---
print("\n" + "=" * 50)
print("Training Final Model on Full Dataset")
print("=" * 50)

best_params = {'learning_rate': 0.01, 'min_child_samples': 20, 'n_estimators': 300, 'num_leaves': 31}
best_lgbm_model = lgb.LGBMRegressor(**best_params, random_state=42)
best_lgbm_model.fit(X, y)

# --- Predict Hours for Next 7 Working Days ---
last_date_in_data = df_209_working_days.index.max()

def find_next_working_days(start_date, n, holiday_calendar):
    next_days = []
    current_date = start_date + timedelta(days=1)
    future_holidays = holiday_calendar.holidays(start=current_date - timedelta(days=365), end=current_date + timedelta(days=365 * 2))
    while len(next_days) < n:
        if current_date.dayofweek < 5 and current_date not in future_holidays:
            next_days.append(current_date)
        current_date += timedelta(days=1)
    return next_days

def get_last_year_values(df, target_date, columns=['Quantity', 'Hours', 'SystemHours']):
    """Get last year same week same day values for prediction"""
    last_year_date = target_date - pd.DateOffset(years=1)
    
    # Create a window of ±7 days around the target date to find working days
    start_window = last_year_date - pd.DateOffset(days=7)
    end_window = last_year_date + pd.DateOffset(days=7)
    
    # Filter data within the window (only working days)
    window_data = df[(df.index >= start_window) & (df.index <= end_window)]
    
    if window_data.empty:
        return {f'{col}_LastYear': np.nan for col in columns}
    
    # Find the date with the same day of week, or closest match
    target_dow = target_date.dayofweek
    window_data_with_dow = window_data.copy()
    window_data_with_dow['dow_diff'] = abs(window_data_with_dow.index.dayofweek - target_dow)
    
    # Sort by day of week difference and then by date difference
    window_data_with_dow['date_diff'] = abs((window_data_with_dow.index - last_year_date).days)
    best_match = window_data_with_dow.sort_values(['dow_diff', 'date_diff']).iloc[0]
    
    # Return the values
    return {f'{col}_LastYear': best_match[col] for col in columns}

cal = SwedishHolidayCalendar()
next_7_working_days = find_next_working_days(last_date_in_data, 7, cal)

future_features_list = []
last_row_data = df_209_working_days.iloc[-1]
future_kpi_value = 293

for pred_date in next_7_working_days:
    future_row = {}
    future_row['month'] = pred_date.month
    future_row['year'] = pred_date.year
    future_row['week_no'] = pred_date.isocalendar().week
    future_row['day_of_week'] = pred_date.dayofweek
    future_row['is_weekend'] = int(pred_date.dayofweek >= 5)
    future_row['quarter'] = pred_date.quarter
    future_row['is_monthend'] = int(pd.to_datetime(pred_date).is_month_end)
    future_row['is_monthstart'] = int(pd.to_datetime(pred_date).is_month_start)
    future_row['ResourceKPI'] = future_kpi_value
    future_row['UtilizationRatio'] = last_row_data['UtilizationRatio']

    # Define lags and windows (re-defined for clarity)
    LAGS = [1, 7, 14, 21, 30]
    WINDOWS = [7, 14]
    COLUMNS_TO_LAG_WINDOW = ['Quantity', 'SystemHours', 'Hours']

    for col in COLUMNS_TO_LAG_WINDOW:
        for lag in LAGS:
            lag_col_name = f'{col}_Lag{lag}'
            future_row[lag_col_name] = last_row_data.get(lag_col_name, np.nan)

        for window in WINDOWS:
            for stat in ['Mean', 'Std', 'Max', 'Min', 'Median', 'Sum', 'EWMA', 'CV', 'IQR']:
                window_col_name = f'{col}_Window{window}_{stat}'
                future_row[window_col_name] = last_row_data.get(window_col_name, np.nan)

    # Get last year same week same day values
    last_year_values = get_last_year_values(df_209_working_days, pred_date)
    future_row.update(last_year_values)

    # Interaction features
    qty_lag7 = future_row.get('Quantity_Lag7', np.nan)
    qty_lag14 = future_row.get('Quantity_Lag14', np.nan)
    qty_last_year = future_row.get('Quantity_LastYear', np.nan)
    
    future_row['ResourceKPI_Quantity_Lag7'] = future_kpi_value * qty_lag7
    future_row['ResourceKPI_Quantity_Lag14'] = future_kpi_value * qty_lag14
    future_row['ResourceKPI_Quantity_LastYear'] = future_kpi_value * qty_last_year

    future_features_list.append(future_row)

X_future = pd.DataFrame(future_features_list, index=next_7_working_days)

# Ensure feature order matches training data (X) and handle potential missing columns in X_future
X_future = X_future.reindex(columns=X.columns, fill_value=np.nan)

predicted_hours = best_lgbm_model.predict(X_future)
predictions = pd.DataFrame({'Predicted_Hours': predicted_hours}, index=next_7_working_days)

print("\nPredicted Hours for the next 7 valid working days:")
print("=" * 50)
for date, pred_hours in predictions.iterrows():
    print(f"{date.strftime('%Y-%m-%d (%A)')}: {pred_hours['Predicted_Hours']:.2f} hours")

print(f"\nTotal predicted hours for next 7 working days: {predicted_hours.sum():.2f} hours")
print(f"Average predicted hours per day: {predicted_hours.mean():.2f} hours")
