
import numpy as np
import pandas as pd
import pickle
from data_preprocessing import preprocess_data

def standard_scaler(X):
    means = X.mean(0)
    stds = X.std(0)
    return (X - means) / stds, means, stds

def sign(x, first_element_zero=False):
    signs = (-1)**(x < 0)
    if first_element_zero:
        signs[0] = 0
    return signs

class RegularizedRegression:
    
    def _record_info(self, X, y, lam, intercept, standardize):
        # Check if the input data has NaNs
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Input data contains NaN values. Ensure preprocessing handles missing values.")
        
        # Standardize the data
        if standardize:
            X, self.means, self.stds = standard_scaler(X)
        else:
            self.means, self.stds = None, None
        
        # Add intercept
        if intercept:
            ones = np.ones(len(X)).reshape(len(X), 1)  # column of ones
            X = np.concatenate((ones, X), axis=1)  # concatenate
            
        # Record values
        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        self.lam = lam
        self.intercept = intercept
        self.standardize = standardize
        
    def fit_ridge(self, X, y, lam=0, intercept=False, standardize=True):
        # Record data and dimensions
        self._record_info(X, y, lam, intercept, standardize)
        
        # Estimate parameters
        XtX = np.dot(self.X.T, self.X)
        I_prime = np.eye(self.D)
        I_prime[0, 0] = 0  # Don't penalize intercept
        XtX_plus_lam_inverse = np.linalg.inv(XtX + self.lam * I_prime)
        Xty = np.dot(self.X.T, self.y)
        self.beta_hats = np.dot(XtX_plus_lam_inverse, Xty)
        
        # Get fitted values
        self.y_hat = np.dot(self.X, self.beta_hats)
        
    def fit_lasso(self, X, y, lam=0, n_iters=2000, lr=0.0001, intercept=False, standardize=True):
        # Record data and dimensions
        self._record_info(X, y, lam, intercept, standardize)
        
        # Estimate parameters using gradient descent
        beta_hats = np.random.randn(self.D)
        for i in range(n_iters):
            dL_dbeta = -self.X.T @ (self.y - (self.X @ beta_hats)) + self.lam * sign(beta_hats, True)
            beta_hats -= lr * dL_dbeta 
        self.beta_hats = beta_hats
        
        # Get fitted values
        self.y_hat = np.dot(self.X, self.beta_hats)

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
        X (np.array): Features
        
        Returns:
        np.array: Predicted values
        """
        if hasattr(self, 'beta_hats'):
            if self.standardize:
                X = (X - self.means) / self.stds
            
            # If intercept was not included during fitting, add it now
            if self.intercept:
                X = np.column_stack((np.ones(X.shape[0]), X))
            
            return np.dot(X, self.beta_hats)
        else:
            raise ValueError("Model has not been fitted yet.")

def k_fold_cross_validation(X, y, k, model_type, lam, n_iters=2000, lr=0.0001):
    """
    Perform k-fold cross-validation.
    
    Args:
    X (np.array): Features
    y (np.array): Target variable
    k (int): Number of folds
    model_type (str): Type of model ('ridge' or 'lasso')
    lam (float): Regularization parameter
    n_iters (int): Number of iterations for Lasso (ignored for Ridge)
    lr (float): Learning rate for Lasso (ignored for Ridge)
    
    Returns:
    float: Mean cross-validation score (R-squared)
    """
    fold_size = len(X) // k
    scores = []
    
    for i in range(k):
        # Create train and validation sets
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i != k - 1 else len(X)
        
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.concatenate([X[:val_start], X[val_end:]])
        y_train = np.concatenate([y[:val_start], y[val_end:]])
        
        # Train model
        model = RegularizedRegression()
        if model_type == 'ridge':
            model.fit_ridge(X_train, y_train, lam=lam, intercept=True, standardize=True)
        elif model_type == 'lasso':
            model.fit_lasso(X_train, y_train, lam=lam, n_iters=n_iters, lr=lr, intercept=True, standardize=True)
        
        # Make predictions and calculate R-squared
        y_pred = model.predict(X_val)
        r2 = 1 - (np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2))
        scores.append(r2)
    
    return np.mean(scores)

def train_model(data_path, target_column, model_type='ridge', lam=0.1, n_iters=2000, lr=0.0001, k=5):
    """
    Main function to train the regularized regression model using k-fold cross-validation.
    
    Args:
    data_path (str): Path to the raw data file
    target_column (str): Name of the target column
    model_type (str): Type of model to train ('ridge' or 'lasso')
    lam (float): Regularization parameter
    n_iters (int): Number of iterations for Lasso (ignored for Ridge)
    lr (float): Learning rate for Lasso (ignored for Ridge)
    k (int): Number of folds for cross-validation
    
    Returns:
    tuple: Trained model and mean cross-validation score
    """
    # Load and preprocess the data
    X, y = preprocess_data(data_path, target_column)
    
    # Check if the dataset contains NaN values after preprocessing
    if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
        raise ValueError("Preprocessed data contains NaN values. Please ensure proper missing value handling.")
    
    # Perform k-fold cross-validation
    try:
        cv_score = k_fold_cross_validation(X.values, y.values, k, model_type, lam, n_iters, lr)
    except ValueError as e:
        print(f"Error during cross-validation: {e}")
        return None
    
    # Train final model on all data
    model = RegularizedRegression()
    if model_type == 'ridge':
        model.fit_ridge(X.values, y.values, lam=lam, intercept=True, standardize=True)
    elif model_type == 'lasso':
        model.fit_lasso(X.values, y.values, lam=lam, n_iters=n_iters, lr=lr, intercept=True, standardize=True)
    else:
        raise ValueError("Invalid model_type. Choose 'ridge' or 'lasso'.")
    
    return model, cv_score

if __name__ == "__main__":
    # Example usage
    data_path = "../data/training_data.csv"
    target_column = "FUEL CONSUMPTION"
    
    # Train Ridge model
    try:
        ridge_model, ridge_cv_score = train_model(data_path, target_column, model_type='ridge', lam=0.1, k=5)
        print(f"Ridge model cross-validation score: {ridge_cv_score:.4f}")
    except Exception as e:
        print(f"Error during Ridge model training: {e}")
    
    # Train Lasso model
    try:
        lasso_model, lasso_cv_score = train_model(data_path, target_column, model_type='lasso', lam=0.1, n_iters=2000, lr=0.0001, k=5)
        print(f"Lasso model cross-validation score: {lasso_cv_score:.4f}")
    except Exception as e:
        print(f"Error during Lasso model training: {e}")
    
    # Save the trained models
    if ridge_model:
        with open('../models/ridge_model_final.pkl', 'wb') as f:
            pickle.dump(ridge_model, f)
    
    if lasso_model:
        with open('../models/lasso_model_final.pkl', 'wb') as f:
            pickle.dump(lasso_model, f)
    
    print("Models trained and saved successfully.")
