import argparse
import pandas as pd
import numpy as np
import pickle
from data_preprocessing import preprocess_data
from train_model import RegularizedRegression


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

def main(model_path, data_path, metrics_output_path, predictions_output_path):
    # Load and preprocess the data
    target_column = "FUEL CONSUMPTION"
    X, y_true = preprocess_data(data_path, target_column)
    
    # Load the model and make predictions
    model = load_model(model_path)
    
    # Check if model is fitted and predict
    try:
        y_pred = model.predict(X.values)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Evaluate the model
    metrics = evaluate_model(y_true, y_pred)
    
    # Save metrics
    with open(metrics_output_path, 'w') as f:
        f.write("Regression Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")  # Consistent rounding
    
    # Optionally, merge predictions with true values for better insights
    predictions_df = pd.DataFrame({
        "True": y_true,
        "Predicted": y_pred
    })
    predictions_df.to_csv(predictions_output_path, index=False)
    
    print("Evaluation complete. Metrics and predictions saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate regression model and generate predictions.")
    parser.add_argument("--model_path", required=True, help="Path to the saved model file")
    parser.add_argument("--data_path", required=True, help="Path to the data CSV file")
    parser.add_argument("--metrics_output_path", required=True, help="Path to save evaluation metrics")
    parser.add_argument("--predictions_output_path", required=True, help="Path to save predictions")
    
    args = parser.parse_args()
    
    main(args.model_path, args.data_path, args.metrics_output_path, args.predictions_output_path)
