"""
Configuration file for the machine learning project.
Contains all paths, hyperparameters and model configurations.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path("./data")
MODEL_DIR = PROJECT_ROOT / "model"

# Data files - Using absolute paths
X_TRAIN_PATH = DATA_DIR / "train_input.csv"
Y_TRAIN_PATH = DATA_DIR / "train_output.csv"
X_TEST_PATH = DATA_DIR / "test_input.csv"

# Model files
MODEL_FREQ_PATH = MODEL_DIR / "model_freq.pkl"
MODEL_CM_PATH = MODEL_DIR / "model_cm.pkl"
COLS_TO_DROP_PATH = MODEL_DIR / "cols_to_drop.pkl"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 1000  # Number of samples per batch
N_JOBS = -1  # Number of parallel jobs (-1 means use all processors)

# Data loading optimization
CHUNK_SIZE = BATCH_SIZE  # Align chunk size with batch size
DTYPE_OPTIMIZATION = True

# Additional parameters
NAN_THRESHOLD = 0.9

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Additional parameters
N_ESTIMATORS = 100

if __name__ == "__main__":
    print(X_TRAIN_PATH)
    print(Y_TRAIN_PATH)
    print(X_TEST_PATH)
    print(MODEL_FREQ_PATH)
    print(MODEL_CM_PATH)
