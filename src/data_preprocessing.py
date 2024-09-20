import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in numeric_columns:
        df[col].fillna(df[col].mean(), inplace=True)
    
    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def target_encoding(df, categorical_columns, target_column, alpha=5):
    """
    Perform target encoding on categorical variables.
    
    Args:
    df (pd.DataFrame): Input dataframe
    categorical_columns (list): List of categorical column names
    target_column (str): Name of the target column
    alpha (float): Smoothing factor
    
    Returns:
    pd.DataFrame: Dataframe with target-encoded features
    """
    global_mean = df[target_column].mean()
    
    for col in categorical_columns:
        # Compute the mean of the target for each category
        category_means = df.groupby(col)[target_column].agg(['mean', 'count'])
        
        # Compute smoothed mean
        smoothed_mean = (category_means['count'] * category_means['mean'] + alpha * global_mean) / (category_means['count'] + alpha)
        
        # Map the smoothed mean back to the original dataframe
        df[f'{col}_encoded'] = df[col].map(smoothed_mean)
        
        # Drop the original categorical column
        df.drop(col, axis=1, inplace=True)
    
    return df

def normalize_features(df):
    """
    Normalize numerical features in the dataset.
    
    Args:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Dataframe with normalized features
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if col != 'FUEL CONSUMPTION':  
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
    
    return df

def preprocess_data(file_path, target_column):
    """
    Main function to preprocess the data.
    
    Args:
    file_path (str): Path to the raw data file
    target_column (str): Name of the target column
    
    Returns:
    tuple: Preprocessed features (X) and target variable (y)
    """
    df = load_data(file_path)
    df = handle_missing_values(df)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = target_encoding(df, categorical_columns, target_column)
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    X = normalize_features(X)
    
    return X, y

if __name__ == "__main__":
    # Example usage right here
    raw_data_path = "data/training_data.csv"
    target_column = "FUEL CONSUMPTION"
    X, y = preprocess_data(raw_data_path, target_column)
    print("Preprocessed data shape:", X.shape)
    print("Target variable shape:", y.shape)