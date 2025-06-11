# test_model_loading.py
import sys
sys.path.append('.')

from utils.data_loader import load_enhanced_models
from config import MODELS_DIR
import os

print(f"MODELS_DIR: {MODELS_DIR}")
print(f"Directory exists: {os.path.exists(MODELS_DIR)}")

if os.path.exists(MODELS_DIR):
    files = os.listdir(MODELS_DIR)
    enhanced_files = [f for f in files if f.startswith('enhanced_model_')]
    print(f"Enhanced model files: {enhanced_files}")

models, metadata, features = load_enhanced_models()
print(f"Loaded models: {list(models.keys()) if models else 'None'}")