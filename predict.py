"""
Prediction script for generating FREQ and CM predictions on test data.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Tuple
import time

from config import (
    X_TEST_PATH,
    N_JOBS
)
from preprocessing import preprocess_test

def load_models() -> Tuple[object, object]:
    """
    Load the trained FREQ and CM models.
    
    Returns:
        Tuple containing (freq_model, cm_model)
    """
    print("Loading models...")
    freq_data = joblib.load('models/xgb_freq_model.joblib')
    cm_data = joblib.load('models/xgb_cm_model.joblib')
    
    return freq_data['model'], cm_data['model']

def generate_predictions(x_test: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for FREQ and CM using the trained models.
    
    Args:
        x_test: Preprocessed test features
        
    Returns:
        DataFrame with FREQ and CM predictions
    """
    # Load models
    freq_model, cm_model = load_models()
    
    print("Generating predictions...")
    # Generate predictions
    freq_pred = freq_model.predict(x_test)
    cm_pred = cm_model.predict(x_test)
    
    # Clip negative values to 0 since FREQ and CM cannot be negative
    freq_pred = np.clip(freq_pred, 0, None)
    cm_pred = np.clip(cm_pred, 0, None)
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        'FREQ': freq_pred,
        'CM': cm_pred
    })
    
    return predictions

def predict():
    """
    Main prediction function.
    """
    print("Starting prediction pipeline...")
    start_time = time.time()
    
    # Load and preprocess test data
    print("Loading and preprocessing test data...")
    x_test_chunks = []
    for chunk in pd.read_csv(X_TEST_PATH, chunksize=10000, low_memory=False):
        x_test_chunks.append(chunk)
    x_test = pd.concat(x_test_chunks, axis=0)
    
    x_test_prep = preprocess_test(x_test)
    print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
    
    # Generate predictions
    predictions = generate_predictions(x_test_prep)
    
    # Save predictions
    output_path = './data/predictions.csv'
    predictions.to_csv(output_path, index=False)
    
    print(f"\nPredictions completed in {time.time() - start_time:.2f} seconds")
    print(f"Predictions saved to: {output_path}")
    print("\nPrediction Summary:")
    print(f"Number of predictions: {len(predictions)}")
    print("\nFREQ predictions:")
    print(f"Mean: {predictions['FREQ'].mean():.4f}")
    print(f"Std: {predictions['FREQ'].std():.4f}")
    print(f"Min: {predictions['FREQ'].min():.4f}")
    print(f"Max: {predictions['FREQ'].max():.4f}")
    print("\nCM predictions:")
    print(f"Mean: {predictions['CM'].mean():.2f}")
    print(f"Std: {predictions['CM'].std():.2f}")
    print(f"Min: {predictions['CM'].min():.2f}")
    print(f"Max: {predictions['CM'].max():.2f}")

if __name__ == "__main__":
    predict() 