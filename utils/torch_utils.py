"""
Utility functions for PyTorch model loading and inference.
This wrapper helps avoid conflicts between Streamlit's file watcher and PyTorch.
"""
import torch
import pickle
import os
import numpy as np
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Define the LSTM model class
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout)
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_torch_models(models_dir):
    """
    Safely load PyTorch models with error handling
    """
    try:
        # Define paths for saved model files
        nn_models_path = os.path.join(models_dir, 'work_utilization_nn_models.pkl')
        nn_scalers_path = os.path.join(models_dir, 'work_utilization_nn_scalers.pkl')
        nn_metrics_path = os.path.join(models_dir, 'work_utilization_nn_metrics.pkl')
        
        # Check if all files exist
        if not all(os.path.exists(path) for path in [nn_models_path, nn_scalers_path, nn_metrics_path]):
            logger.warning("One or more neural network model files not found")
            return {}, {}, {}
        
        # Load models
        with open(nn_models_path, 'rb') as f:
            nn_models = pickle.load(f)
        
        # Load scalers
        with open(nn_scalers_path, 'rb') as f:
            nn_scalers = pickle.load(f)
        
        # Load metrics
        with open(nn_metrics_path, 'rb') as f:
            nn_metrics = pickle.load(f)
        
        # Make sure models are in evaluation mode
        for model_name, model in nn_models.items():
            model.eval()
        
        logger.info(f"Neural network models loaded successfully. Number of models: {len(nn_models)}")
        return nn_models, nn_scalers, nn_metrics
    
    except Exception as e:
        logger.error(f"Error loading neural network models: {str(e)}")
        return {}, {}, {}

def predict_with_torch_model(model, scaler, features, sequence_length=7):
    """
    Make a prediction using a PyTorch LSTM model
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained LSTM model
    scaler : sklearn.preprocessing.MinMaxScaler
        The scaler used to normalize the data
    features : numpy.ndarray
        The features to use for prediction
    sequence_length : int
        The length of the sequence for the LSTM model
    
    Returns:
    --------
    float
        The predicted value
    """
    try:
        # Ensure model is in evaluation mode
        model.eval()
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Get the sequence (last sequence_length rows)
        sequence = features_scaled[-sequence_length:, 1:]  # Exclude the target column
        
        # Convert to tensor
        sequence_tensor = torch.tensor(sequence.reshape(1, sequence_length, -1), dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            output = model(sequence_tensor)
            prediction_scaled = output.numpy()[0][0]
        
        # Inverse transform to get the actual prediction
        prediction_full = np.zeros((1, scaler.n_features_in_))
        prediction_full[0, 0] = prediction_scaled
        prediction = scaler.inverse_transform(prediction_full)[0][0]
        
        # Ensure prediction is not negative
        prediction = max(0, prediction)
        
        return prediction
    
    except Exception as e:
        logger.error(f"Error making prediction with neural network: {str(e)}")
        return None