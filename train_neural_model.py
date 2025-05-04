"""
Train neural network model for workforce prediction using PyTorch with improved validation
"""
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.data_loader import load_data
from utils.feature_engineering import engineer_features, create_lag_features
from config import MODELS_DIR

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Define PyTorch LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Custom dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# Modified MAPE with minimum threshold
def modified_mape(y_true, y_pred, epsilon=1.0):
    """Calculate MAPE with a minimum threshold to avoid division by zero"""
    # Use max of actual value or epsilon as denominator
    denominator = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs(y_pred - y_true) / denominator) * 100


def prepare_sequence_data(df, work_type, sequence_length=7):
    """
    Prepare sequence data for LSTM model with all available features
    """
    # Filter data for the specific work type
    data = df[df['WorkType'] == work_type].copy()
    
    # Sort by date
    data = data.sort_values('Date')
    
    # Define features to use - same as in the RandomForest model
    numeric_features = [
        'NoOfMan_lag_1', 'NoOfMan_lag_2', 'NoOfMan_lag_3', 'NoOfMan_lag_7', 
        'NoOfMan_lag_14', 'NoOfMan_lag_30', 'NoOfMan_rolling_mean_7',
        'NoOfMan_rolling_max_7', 'NoOfMan_rolling_min_7', 'NoOfMan_rolling_std_7',
        'NoOfMan_same_dow_lag', 'IsWeekend_feat'
    ]
    
    categorical_features = ['DayOfWeek_feat', 'Month_feat']
    
    # Select target column first (for easier extraction later)
    all_features = ['NoOfMan'] + numeric_features + categorical_features
    
    # Select only columns we need
    data = data[all_features].values
    
    # Scale the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length, 1:])  # All features except NoOfMan
        y.append(data_scaled[i+sequence_length, 0])     # NoOfMan is the target (first column)
    
    return np.array(X), np.array(y), scaler

def train_neural_models(ts_data, work_types):
    """
    Train neural network models for each work type using TimeSeriesSplit
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    nn_models = {}
    nn_scalers = {}
    nn_metrics = {}
    
    for work_type in work_types:
        print(f"Training neural network for WorkType: {work_type}")
        
        # Filter data for this WorkType
        work_type_data = ts_data[ts_data['WorkType'] == work_type]
        
        if len(work_type_data) < 100:  # Skip if not enough data
            print(f"  Skipping {work_type}: Not enough data ({len(work_type_data)} records)")
            continue
        
        # Prepare sequence data
        X, y, scaler = prepare_sequence_data(work_type_data, work_type)
        
        if len(X) < 50:  # Additional check after sequence preparation
            print(f"  Skipping {work_type}: Not enough sequences ({len(X)} sequences)")
            continue
        
        # Setup TimeSeriesSplit for cross-validation
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Initialize metrics storage for cross-validation
        cv_metrics = {
            'MAE': [],
            'RMSE': [],
            'R²': [],
            'MAPE': []
        }
        
        fold = 1
        best_model = None
        best_r2 = -float('inf')
        
        # Perform cross-validation
        for train_idx, test_idx in tscv.split(X):
            print(f"  Training fold {fold}/{n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create PyTorch datasets and dataloaders
            train_dataset = TimeSeriesDataset(X_train, y_train.reshape(-1, 1))
            test_dataset = TimeSeriesDataset(X_test, y_test.reshape(-1, 1))
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            input_size = X.shape[2]
            model = LSTMModel(input_size=input_size).to(device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            num_epochs = 100
            best_loss = float('inf')
            patience = 20
            counter = 0
            fold_best_model = None
            
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    fold_best_model = model.state_dict().copy()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"    Early stopping at epoch {epoch}")
                        break
                
                if epoch % 10 == 0:
                    print(f"    Epoch {epoch}, Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(test_loader):.4f}")
            
            # Load best model for this fold
            model.load_state_dict(fold_best_model)
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                y_pred = []
                for batch_X, _ in test_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    y_pred.extend(outputs.cpu().numpy())
            
            y_pred = np.array(y_pred).flatten()
            
            # Transform predictions back to original scale for metrics
            y_pred_orig = np.zeros((len(y_pred), scaler.n_features_in_))
            y_pred_orig[:, 0] = y_pred
            y_pred_orig = scaler.inverse_transform(y_pred_orig)[:, 0]
            
            y_test_orig = np.zeros((len(y_test), scaler.n_features_in_))
            y_test_orig[:, 0] = y_test
            y_test_orig = scaler.inverse_transform(y_test_orig)[:, 0]
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_orig, y_pred_orig)
            rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
            r2 = r2_score(y_test_orig, y_pred_orig)
            
            # Calculate modified MAPE
            mape = modified_mape(y_test_orig, y_pred_orig, epsilon=1.0)
            
            # Store metrics for this fold
            cv_metrics['MAE'].append(mae)
            cv_metrics['RMSE'].append(rmse)
            cv_metrics['R²'].append(r2)
            cv_metrics['MAPE'].append(mape)
            
            print(f"    Fold {fold} metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
            
            # Keep the best model across folds
            if r2 > best_r2:
                best_r2 = r2
                best_model = fold_best_model.copy()
            
            fold += 1
        
        # Calculate average metrics across folds
        avg_metrics = {metric: np.mean(values) for metric, values in cv_metrics.items()}
        
        # Print cross-validation details
        print(f"  Cross-validation metrics for {work_type}:")
        for i in range(n_splits):
            print(f"    Fold {i+1}: MAE={cv_metrics['MAE'][i]:.4f}, RMSE={cv_metrics['RMSE'][i]:.4f}, "
                  f"R²={cv_metrics['R²'][i]:.4f}, MAPE={cv_metrics['MAPE'][i]:.2f}%")
        
        print(f"  Average metrics - MAE: {avg_metrics['MAE']:.4f}, RMSE: {avg_metrics['RMSE']:.4f}, "
              f"R²: {avg_metrics['R²']:.4f}, MAPE: {avg_metrics['MAPE']:.2f}%")
        
        # Train final model on all data
        print(f"  Training final model on all data")
        
        final_dataset = TimeSeriesDataset(X, y.reshape(-1, 1))
        final_loader = DataLoader(final_dataset, batch_size=32, shuffle=True)
        
        final_model = LSTMModel(input_size=X.shape[2]).to(device)
        
        if best_model is not None:
            # Use best model from cross-validation as starting point
            final_model.load_state_dict(best_model)
        
        optimizer = optim.Adam(final_model.parameters(), lr=0.0005)  # Lower learning rate for fine-tuning
        
        # Fine-tune on all data
        num_epochs = 50
        for epoch in range(num_epochs):
            final_model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in final_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = final_model(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"    Final training epoch {epoch}, Loss: {epoch_loss/len(final_loader):.4f}")
        
        # Save model and scaler
        nn_models[work_type] = final_model.to('cpu')
        nn_scalers[work_type] = scaler
        nn_metrics[work_type] = avg_metrics
    
    return nn_models, nn_scalers, nn_metrics

def main(file_path):
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(file_path)
    
    # Engineer features
    print("Engineering features...")
    processed_df = engineer_features(df)
    
    # Get list of unique WorkTypes
    work_types = processed_df['WorkType'].unique()
    print(f"Found {len(work_types)} unique WorkTypes")
    
    # Create lag features
    print("Creating lag features...")
    ts_data = create_lag_features(processed_df, 'WorkType', 'NoOfMan')
    
    # Train neural network models
    print("Training neural network models...")
    nn_models, nn_scalers, nn_metrics = train_neural_models(ts_data, work_types)
    
    # Save neural network models, scalers, and metrics
    print("Saving neural network model files...")
    with open(os.path.join(MODELS_DIR, "work_utilization_nn_models.pkl"), "wb") as f:
        pickle.dump(nn_models, f)
    
    with open(os.path.join(MODELS_DIR, "work_utilization_nn_scalers.pkl"), "wb") as f:
        pickle.dump(nn_scalers, f)
    
    with open(os.path.join(MODELS_DIR, "work_utilization_nn_metrics.pkl"), "wb") as f:
        pickle.dump(nn_metrics, f)
    
    print(f"Successfully saved {len(nn_models)} neural network models!")
    return nn_models, nn_scalers, nn_metrics

if __name__ == "__main__":
    # Replace with your file path
    file_path = "C:/forlogssystems/Data/work_utilization_melted1.xlsx"
    nn_models, nn_scalers, nn_metrics = main(file_path)

import sys
sys.modules['__main__'].LSTMModel = LSTMModel