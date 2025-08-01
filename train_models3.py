import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import logging
import traceback
import argparse
import json
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit

from utils.feature_engineering import EnhancedFeatureTransformer
from utils.sql_data_connector import extract_sql_data
from config import ENHANCED_WORK_TYPES, MODELS_DIR, SQL_SERVER, SQL_DATABASE, SQL_TRUSTED_CONNECTION

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "lstm_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lstm_train_models")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# LSTM Hyperparameters
LSTM_CONFIG = {
    'sequence_length': 14,  # Use 14 days of history to predict next day
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 200,
    'early_stopping_patience': 20,
    'min_samples_per_worktype': 100  # Minimum samples needed for LSTM training
}

class WorkforceLSTM(nn.Module):
    """
    LSTM model for workforce prediction
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(WorkforceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        output = self.dropout(last_output)
        
        # Apply linear layer
        output = self.linear(output)
        
        return output

def load_training_data():
    """Load training data same as train_models2.py"""
    try:
        logger.info(f"Loading training data for LSTM models {ENHANCED_WORK_TYPES}")
        query = """
        SELECT Date, PunchCode as WorkType, Hours, SystemHours, 
        CASE WHEN PunchCode IN (206, 213) THEN NoRows
        ELSE Quantity END as Quantity
        FROM WorkUtilizationData 
        WHERE PunchCode IN ('202', '203', '206', '209', '210', '211', '213', '214', '215', '217') 
        AND Hours > 0 
        AND SystemHours > 0 
        AND NoRows > 0
        AND Date < '2025-05-06'
        ORDER BY Date
        """
        df = extract_sql_data(
            server=SQL_SERVER,
            database=SQL_DATABASE,
            query=query,
            trusted_connection=SQL_TRUSTED_CONNECTION
        )
        if df is None or df.empty:
            logger.error("No data returned from SQL query")
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        df['WorkType'] = df['WorkType'].astype(str)
        logger.info(f"Loaded {len(df)} records for LSTM training")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_sequences(df, sequence_length=14):
    """
    Create sequences for LSTM training
    """
    try:
        # Apply feature engineering
        feature_transformer = EnhancedFeatureTransformer()
        features_df = feature_transformer.fit_transform(df)
        
        # Remove non-numeric columns for LSTM
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_columns]
        
        # Handle missing values
        features_df = features_df.fillna(features_df.mean())
        
        sequences = []
        targets = []
        dates = []
        
        # Create sequences for each record (if we have enough history)
        for i in range(sequence_length, len(features_df)):
            # Get sequence of features (past sequence_length days)
            sequence = features_df.iloc[i-sequence_length:i].values
            
            # Target is log-transformed Hours (same as train_models2.py)
            target = np.log1p(df.iloc[i]['Hours'])
            
            sequences.append(sequence)
            targets.append(target)
            dates.append(df.iloc[i]['Date'])
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Created {len(sequences)} sequences with shape {sequences.shape}")
        logger.info(f"Feature dimension: {sequences.shape[2]}")
        
        return sequences, targets, dates, features_df.columns.tolist()
        
    except Exception as e:
        logger.error(f"Error creating sequences: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None

def train_lstm_model(df, work_type):
    """
    Train LSTM model for a specific work type
    """
    try:
        logger.info(f"Training LSTM model for WorkType {work_type}")
        
        # Filter data for this work type
        work_data = df[df['WorkType'] == work_type].copy()
        work_data = work_data.sort_values('Date').reset_index(drop=True)
        
        # Check if we have enough data
        if len(work_data) < LSTM_CONFIG['min_samples_per_worktype']:
            logger.warning(f"Insufficient data for {work_type}: {len(work_data)} samples")
            return None, None, None
        
        # Create sequences
        sequences, targets, dates, feature_names = create_sequences(
            work_data, LSTM_CONFIG['sequence_length']
        )
        
        if sequences is None:
            logger.error(f"Failed to create sequences for {work_type}")
            return None, None, None
        
        # Split data temporally (same approach as train_models2.py)
        test_size = int(len(sequences) * 0.2)
        
        X_train_cv = sequences[:-test_size]
        y_train_cv = targets[:-test_size]
        X_test_final = sequences[-test_size:]
        y_test_final = targets[-test_size:]
        dates_test = dates[-test_size:]
        
        logger.info(f"Data split - Train/CV: {len(X_train_cv)} samples, Final Test: {len(X_test_final)} samples")
        
        # Scale features
        scaler = StandardScaler()
        # Reshape for scaling (combine batch and sequence dimensions)
        X_train_reshaped = X_train_cv.reshape(-1, X_train_cv.shape[2])
        scaler.fit(X_train_reshaped)
        
        # Apply scaling
        X_train_scaled = np.array([
            scaler.transform(seq) for seq in X_train_cv
        ])
        X_test_scaled = np.array([
            scaler.transform(seq) for seq in X_test_final
        ])
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=3)  # Fewer splits due to sequence requirement
        fold_scores = []
        best_val_loss = float('inf')
        best_model_state = None
        
        input_size = X_train_scaled.shape[2]
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
            logger.info(f"Training fold {fold + 1}/3")
            
            X_fold_train = X_train_scaled[train_idx]
            y_fold_train = y_train_cv[train_idx]
            X_fold_val = X_train_scaled[val_idx]
            y_fold_val = y_train_cv[val_idx]
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_fold_train),
                torch.FloatTensor(y_fold_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_fold_val),
                torch.FloatTensor(y_fold_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=LSTM_CONFIG['batch_size'], shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=LSTM_CONFIG['batch_size'], shuffle=False)
            
            # Initialize model
            model = WorkforceLSTM(
                input_size=input_size,
                hidden_size=LSTM_CONFIG['hidden_size'],
                num_layers=LSTM_CONFIG['num_layers'],
                dropout=LSTM_CONFIG['dropout']
            )
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=LSTM_CONFIG['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop
            best_val_loss_fold = float('inf')
            patience_counter = 0
            
            for epoch in range(LSTM_CONFIG['epochs']):
                # Training
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        val_predictions.extend(outputs.numpy())
                        val_targets.extend(batch_y.numpy())
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss_fold:
                    best_val_loss_fold = avg_val_loss
                    patience_counter = 0
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= LSTM_CONFIG['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
            
            # Calculate fold metrics (inverse transform predictions)
            val_pred_hours = np.expm1(val_predictions)
            val_true_hours = np.expm1(val_targets)
            
            fold_mae = mean_absolute_error(val_true_hours, val_pred_hours)
            fold_r2 = r2_score(val_true_hours, val_pred_hours)
            fold_rmse = np.sqrt(mean_squared_error(val_true_hours, val_pred_hours))
            fold_mape = np.mean(np.abs((val_true_hours - val_pred_hours) / np.maximum(val_true_hours, 10))) * 100
            
            fold_scores.append({
                'MAE': fold_mae,
                'RMSE': fold_rmse,
                'R2': fold_r2,
                'MAPE': fold_mape
            })
            
            logger.info(f"Fold {fold + 1} - MAE: {fold_mae:.3f}, R²: {fold_r2:.3f}, MAPE: {fold_mape:.2f}%")
        
        # Train final model on all training data with best hyperparameters
        logger.info("Training final model on all training data...")
        
        final_model = WorkforceLSTM(
            input_size=input_size,
            hidden_size=LSTM_CONFIG['hidden_size'],
            num_layers=LSTM_CONFIG['num_layers'],
            dropout=LSTM_CONFIG['dropout']
        )
        
        if best_model_state:
            final_model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        final_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            test_predictions_log = final_model(X_test_tensor).squeeze().numpy()
            
        # Inverse transform predictions
        test_predictions = np.expm1(test_predictions_log)
        test_targets = np.expm1(y_test_final)
        
        # Calculate test metrics
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_r2 = r2_score(test_targets, test_predictions)
        test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
        test_mape = np.mean(np.abs((test_targets - test_predictions) / np.maximum(test_targets, 10))) * 100
        
        # Calculate average CV metrics
        avg_cv_mae = np.mean([score['MAE'] for score in fold_scores])
        avg_cv_r2 = np.mean([score['R2'] for score in fold_scores])
        avg_cv_mape = np.mean([score['MAPE'] for score in fold_scores])
        
        # Plot test predictions
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(test_targets, label='Actual', alpha=0.7)
        plt.plot(test_predictions, label='Predicted', alpha=0.7)
        plt.legend()
        plt.title(f"LSTM Test Predictions - {work_type}")
        plt.xlabel('Time')
        plt.ylabel('Hours')
        
        plt.subplot(1, 3, 2)
        plt.scatter(test_targets, test_predictions, alpha=0.6)
        plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], 'r--')
        plt.xlabel('Actual Hours')
        plt.ylabel('Predicted Hours')
        plt.title(f"Scatter Plot - R²={test_r2:.3f}")
        
        plt.subplot(1, 3, 3)
        residuals = test_targets - test_predictions
        plt.scatter(test_predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Hours')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f"lstm_test_predictions_{work_type}.png"))
        plt.close()
        
        # Create metadata
        model_metadata = {
            'work_type': work_type,
            'model_type': 'LSTM',
            'training_records': len(X_train_cv),
            'test_records': len(X_test_final),
            'sequence_length': LSTM_CONFIG['sequence_length'],
            'test_period': {
                'start': str(dates_test[0]) if dates_test else None,
                'end': str(dates_test[-1]) if dates_test else None
            },
            # Test set metrics
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            # Cross-validation metrics
            'cv_mae': avg_cv_mae,
            'cv_r2': avg_cv_r2,
            'cv_mape': avg_cv_mape,
            'cv_folds': len(fold_scores),
            # Model configuration
            'lstm_config': LSTM_CONFIG,
            'feature_names': feature_names,
            'input_size': input_size,
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        }
        
        logger.info(f"✅ LSTM model trained for {work_type}")
        logger.info(f"   Test MAE: {test_mae:.3f} (on {len(X_test_final)} samples)")
        logger.info(f"   Test R²: {test_r2:.3f}")
        logger.info(f"   Test MAPE: {test_mape:.2f}%")
        logger.info(f"   CV MAE: {avg_cv_mae:.3f} (avg of {len(fold_scores)} folds)")
        logger.info(f"   CV R²: {avg_cv_r2:.3f}")
        
        return final_model, scaler, model_metadata
        
    except Exception as e:
        logger.error(f"Error training LSTM model for {work_type}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def save_lstm_models(models, scalers, metadata, df):
    """
    Save LSTM models, scalers, and metadata
    """
    try:
        logger.info("Saving LSTM models and metadata")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for work_type in models.keys():
            if models[work_type] is not None:
                # Save PyTorch model
                model_filename = f"lstm_model_{work_type}.pth"
                model_path = os.path.join(MODELS_DIR, model_filename)
                torch.save({
                    'model_state_dict': models[work_type].state_dict(),
                    'model_config': {
                        'input_size': metadata[work_type]['input_size'],
                        'hidden_size': LSTM_CONFIG['hidden_size'],
                        'num_layers': LSTM_CONFIG['num_layers'],
                        'dropout': LSTM_CONFIG['dropout']
                    }
                }, model_path)
                
                # Save scaler
                scaler_filename = f"lstm_scaler_{work_type}.pkl"
                scaler_path = os.path.join(MODELS_DIR, scaler_filename)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scalers[work_type], f)
                
                logger.info(f"  ✅ Saved LSTM model and scaler for {work_type}")
        
        # Save metadata
        metadata_filename = f"lstm_models_metadata_{timestamp}.json"
        metadata_path = os.path.join(MODELS_DIR, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save training data for reference
        try:
            training_data_path = os.path.join(MODELS_DIR, 'lstm_training_data.pkl')
            df.to_pickle(training_data_path)
            logger.info(f"✅ LSTM training data saved: {training_data_path}")
        except Exception as e:
            logger.error(f"⚠️ Failed to save training data: {str(e)}")
        
        logger.info(f"✅ All LSTM models and metadata saved")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving LSTM models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Main training function
    """
    parser = argparse.ArgumentParser(description='Train LSTM workforce prediction models')
    parser.add_argument('--punch-code', type=str, help='Train specific punch code (e.g., 206)')
    parser.add_argument('--all', action='store_true', help='Train all punch codes (default behavior)')
    args = parser.parse_args()
    
    try:
        # Check PyTorch availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Training will use CPU (slower).")
        else:
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        
        df = load_training_data()
        if df is None:
            logger.error("❌ Failed to load training data. Exiting.")
            return
        
        # Determine which work types to process
        if args.punch_code:
            if args.punch_code not in ENHANCED_WORK_TYPES:
                logger.error(f"❌ Punch code {args.punch_code} not in enhanced work types: {ENHANCED_WORK_TYPES}")
                return
            if args.punch_code not in df['WorkType'].unique():
                logger.error(f"❌ No data available for punch code {args.punch_code}")
                logger.info(f"Available work types in data: {list(df['WorkType'].unique())}")
                return
            work_types_to_process = [args.punch_code]
            logger.info(f"🎯 Training single LSTM model for punch code: {args.punch_code}")
        else:
            work_types_to_process = [wt for wt in df['WorkType'].unique() if wt in ENHANCED_WORK_TYPES]
            logger.info(f"🎯 Training LSTM models for all available punch codes: {work_types_to_process}")
        
        logger.info("📊 Data distribution:")
        for work_type in work_types_to_process:
            if work_type in df['WorkType'].unique():
                wt_data = df[df['WorkType'] == work_type]
                logger.info(f"  WorkType {work_type}: {len(wt_data)} records")
                logger.info(f"    Date range: {wt_data['Date'].min()} to {wt_data['Date'].max()}")
                logger.info(f"    Hours avg: {wt_data['Hours'].mean():.2f}")
        
        models = {}
        scalers = {}
        metadata = {}
        
        for work_type in work_types_to_process:
            if work_type not in df['WorkType'].unique():
                logger.warning(f"⚠️ No data available for punch code {work_type}")
                continue
            
            logger.info(f"\n🎯 Processing WorkType {work_type} with LSTM")
            work_data = df[df['WorkType'] == work_type].copy()
            
            if len(work_data) < LSTM_CONFIG['min_samples_per_worktype']:
                logger.warning(f"Skipping {work_type}: Insufficient data ({len(work_data)} < {LSTM_CONFIG['min_samples_per_worktype']} required)")
                continue
            
            model, scaler, model_metadata = train_lstm_model(df, work_type)
            if model is not None:
                models[work_type] = model
                scalers[work_type] = scaler
                metadata[work_type] = model_metadata
                logger.info(f"✅ Successfully trained LSTM model for {work_type}")
            else:
                logger.error(f"❌ Failed to train LSTM model for {work_type}")
        
        if models:
            success = save_lstm_models(models, scalers, metadata, df)
            if success:
                logger.info("\n🎉 LSTM MODEL TRAINING COMPLETED SUCCESSFULLY")
                logger.info("=" * 60)
                logger.info(f"✅ Trained LSTM models: {list(models.keys())}")
                
                # Compare with LightGBM results if available
                logger.info("\n📊 LSTM vs LightGBM COMPARISON:")
                logger.info("-" * 40)
                for work_type, meta in metadata.items():
                    logger.info(f"\n📈 {work_type} LSTM Performance:")
                    logger.info(f"   Test MAE: {meta['test_mae']:.3f}")
                    logger.info(f"   Test R²: {meta['test_r2']:.3f}")
                    logger.info(f"   Test MAPE: {meta['test_mape']:.2f}%")
                    logger.info(f"   CV MAE: {meta['cv_mae']:.3f}")
                    logger.info(f"   CV R²: {meta['cv_r2']:.3f}")
                    logger.info(f"   Sequence Length: {meta['sequence_length']} days")
                
                logger.info("\n💡 USAGE NOTES:")
                logger.info("   - Compare these metrics with train_models2.py (LightGBM) results")
                logger.info("   - LSTM models are saved as .pth files with separate scalers")
                logger.info("   - Integration with prediction system requires additional work")
                logger.info("   - Consider ensemble approach if both models perform well")
                
            else:
                logger.error("❌ Failed to save LSTM models")
        else:
            logger.error("❌ No LSTM models were successfully trained")
            logger.info("💡 Try reducing min_samples_per_worktype in LSTM_CONFIG if datasets are small")
    
    except Exception as e:
        logger.error(f"❌ Error in main LSTM training process: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()