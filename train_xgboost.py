"""
XGBoost training module.
Contains functions for training XGBoost models for frequency and CM prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import time
from typing import Tuple, Dict
import joblib

from config import (
    TEST_SIZE, RANDOM_STATE, 
    X_TRAIN_PATH, Y_TRAIN_PATH,
    N_JOBS
)
from preprocessing import load_data, preprocess_train

def train_xgb_model(
    X: pd.DataFrame, 
    y: pd.Series,
    model_name: str
) -> Tuple[xgb.XGBRegressor, Dict]:
    """
    Train XGBoost model and return metrics.
    
    Args:
        X: Training features
        y: Target variable
        model_name: Name of the model for logging
        
    Returns:
        Trained model and dictionary of metrics
    """
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Initialize XGBoost model with reasonable defaults
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method='hist',  # For faster training
        enable_categorical=True  # Enable categorical feature support
    )
    
    print(f"\nTraining {model_name} model...")
    start_time = time.time()
    
    # Train the model
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Calculate metrics
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
        'train_mae': mean_absolute_error(y_train, train_pred),
        'val_mae': mean_absolute_error(y_val, val_pred),
        'train_r2': r2_score(y_train, train_pred),
        'val_r2': r2_score(y_val, val_pred),
        'training_time': time.time() - start_time
    }
    
    print(f"\n{model_name} Training Metrics:")
    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
    print(f"Validation RMSE: {metrics['val_rmse']:.4f}")
    print(f"Training R²: {metrics['train_r2']:.4f}")
    print(f"Validation R²: {metrics['val_r2']:.4f}")
    print(f"Training time: {metrics['training_time']:.2f} seconds")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    print(f"\nTop 10 most important features for {model_name}:")
    print(importance.head(10))
    
    return model, metrics

def train():
    """
    Main training function for XGBoost models.
    """
    print("Loading and preprocessing data...")
    start_time = time.time()
    
    # Load and preprocess data
    x_train, y_train, _ = load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TRAIN_PATH)
    x_train_prep = preprocess_train(x_train)
    
    print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
    
    # Train FREQ model
    freq_model, freq_metrics = train_xgb_model(
        x_train_prep, 
        y_train['FREQ'],
        "FREQ"
    )
    
    # Train CM model
    cm_model, cm_metrics = train_xgb_model(
        x_train_prep, 
        y_train['CM'],
        "CM"
    )
    
    # Save models and metrics
    joblib.dump({
        'model': freq_model,
        'metrics': freq_metrics
    }, 'models/xgb_freq_model.joblib')
    
    joblib.dump({
        'model': cm_model,
        'metrics': cm_metrics
    }, 'models/xgb_cm_model.joblib')
    
    print('\nModels trained and saved successfully.')

if __name__ == "__main__":
    train() 