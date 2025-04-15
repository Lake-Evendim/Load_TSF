import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(data):
    """
    Clean the dataset by handling missing values and duplicates.
    """
    # Drop duplicates
    data = data.drop_duplicates()
    
    # Fill missing values with mean for numerical columns
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col].fillna(data[col].mean(), inplace=True)
    
    # Fill missing values with mode for categorical columns
    for col in data.select_dtypes(include=[object]).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)
    
    print("Data cleaned successfully.")
    return data

def preprocess_data(data, target_column):
    """
    Preprocess the dataset by splitting and scaling.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    file_path = "data.csv"  # Replace with your dataset path
    target_column = "target"  # Replace with your target column name
    
    data = load_data(file_path)
    if data is not None:
        data = clean_data(data)
        X_train, X_test, y_train, y_test = preprocess_data(data, target_column)