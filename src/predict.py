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