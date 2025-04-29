import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def basic_info(data):
    """
    Print basic information about the dataset.
    """
    print("Dataset Info:")
    print(data.info())
    print("\nFirst 5 Rows:")
    print(data.head())
    print("\nBasic Statistics:")
    print(data.describe())

def check_missing_values(data):
    """
    Check for missing values in the dataset.
    """
    missing = data.isnull().sum()
    print("\nMissing Values:")
    print(missing[missing > 0])

def check_duplicates(data):
    """
    Check for duplicate rows in the dataset.
    """
    duplicates = data.duplicated().sum()
    print(f"\nNumber of Duplicate Rows: {duplicates}")

def plot_distributions(data, columns):
    """
    Plot the distribution of specified columns.
    """
    for column in columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

def plot_correlations(data, target_columns):
    """
    Plot the correlation matrix for the dataset.
    """
    plt.figure(figsize=(12, 8))
    corr_matrix = data[target_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_time_series(data, time_column, target_columns):
    """
    Plot time series for the specified target columns.
    """
    for column in target_columns:
        plt.figure(figsize=(12, 6))
        plt.plot(data[time_column], data[column])
        plt.title(f"Time Series of {column}")
        plt.xlabel("Time")
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()

def eda_pipeline(file_path, time_column, target_columns):
    """
    Full EDA pipeline:
    1. Load data
    2. Basic info
    3. Check missing values
    4. Check duplicates
    5. Plot distributions
    6. Plot correlations
    7. Plot time series
    """
    data = load_data(file_path)
    basic_info(data)
    check_missing_values(data)
    check_duplicates(data)
    plot_distributions(data, target_columns)
    plot_correlations(data, target_columns)
    plot_time_series(data, time_column, target_columns)