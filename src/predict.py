import argparse
import pandas as pd
import numpy as np
import pickle
from data_preprocessing import preprocess_data

def load_model(model_path):
    """
    Load the trained model from a file.
    
    Args:
    model_path (str): Path to the saved model file
    
    Returns:
    RegularizedRegression: Loaded regression model
    """
    with open(model_path, 'rb') as f:
        return pickle.load(f)
    

def evaluate_model(y_true, y_pred):
    """
    Calculate evaluation metrics for the model.
    
    Args:
    y_true (np.array): True target values
    y_pred (np.array): Predicted target values
    
    Returns:
    dict: Dictionary containing evaluation metrics
    """
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    return {
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'R-squared (R²) Score': r2
    }
