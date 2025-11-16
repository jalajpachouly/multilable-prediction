"""Data loading and preprocessing functions."""
import pandas as pd
import numpy as np
from scipy.optimize import nnls
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from typing import Tuple


def build_conditional_prob_matrix(df: pd.DataFrame, labels: list) -> np.ndarray:
    """
    Build a conditional probability matrix for label co-occurrence.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the labels.
    - labels (list of str): List of label column names.

    Returns:
    - cooc_norm (np.ndarray): Normalized co-occurrence matrix.
    """
    cooc = df[labels].values.T.dot(df[labels].values)
    cooc_norm = cooc.copy().astype(np.float32)
    for i in range(cooc_norm.shape[0]):
        cooc_norm[:, i] /= cooc[i, i]
    return cooc_norm


def nnls_sample(df: pd.DataFrame, labels: list, target_count: int, cond_prob: np.ndarray) -> pd.DataFrame:
    """
    Perform stratified sampling to balance the dataset based on label co-occurrence.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data and labels.
    - labels (list of str): List of label column names.
    - target_count (int): Desired number of samples per label.
    - cond_prob (np.ndarray): Conditional probability matrix from build_conditional_prob_matrix.

    Returns:
    - sampled_df (pd.DataFrame): The resampled DataFrame.
    """
    target_counts = np.array([target_count for _ in labels])
    optimal_samples, residuals = nnls(cond_prob, target_counts)
    optimal_samples = np.ceil(optimal_samples).astype(np.int32)
    df_subs = []
    for i, label in enumerate(labels):
        sub_df = df[df[label] == 1]
        df_subs.append(sub_df.sample(optimal_samples[i],
                                     replace=len(sub_df) < optimal_samples[i]))
    sampled_df = pd.concat(df_subs)
    return sampled_df


def load_data(csv_path: str, labels: list, sample_size: int = None, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset from a CSV file and split it into training and testing sets using stratified splitting.

    Parameters:
    - csv_path (str): Path to the CSV file containing the dataset.
    - labels (list): List of label column names.
    - sample_size (int, optional): Number of rows to sample from dataset (for faster testing). None means use all data.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train (pd.DataFrame): Training set features.
    - X_test (pd.DataFrame): Testing set features.
    - y_train (pd.DataFrame): Training set labels.
    - y_test (pd.DataFrame): Testing set labels.
    """
    # Load the dataset from a CSV file
    data = pd.read_csv(csv_path)

    # Sample data if requested
    if sample_size is not None and sample_size > 0 and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=random_state)
        print(f"[DATA] Sampled {sample_size} rows from dataset for faster processing")

    # Check if required columns exist
    required_columns = ['report'] + labels
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

    # Feature column (text data)
    X = data[['report']]

    # Label columns (multi-label targets)
    y = data[labels]

    # Initialize the stratified shuffle split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

    # Perform the split
    for train_index, test_index in msss.split(X, y):
        X_train = X.iloc[train_index].reset_index(drop=True)
        X_test = X.iloc[test_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)

    # Print label counts to check for class imbalance
    print("Label counts in y_train:")
    print(y_train.sum())
    print("\nLabel counts in y_test:")
    print(y_test.sum())
    return X_train, X_test, y_train, y_test


def load_data_balanced(csv_path: str, labels: list, target_count: int = 600, sample_size: int = None, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset from a CSV file, balance it, and split into training and testing sets.

    Parameters:
    - csv_path (str): Path to the CSV file containing the dataset.
    - labels (list): List of label column names.
    - target_count (int): Desired number of samples per label for balancing.
    - sample_size (int, optional): Number of rows to sample from dataset before balancing (for faster testing). None means use all data.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train (pd.DataFrame): Training set features.
    - X_test (pd.DataFrame): Testing set features.
    - y_train (pd.DataFrame): Training set labels.
    - y_test (pd.DataFrame): Testing set labels.
    """
    # Load the dataset from a CSV file
    data = pd.read_csv(csv_path)

    # Sample data if requested (before balancing)
    if sample_size is not None and sample_size > 0 and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=random_state)
        print(f"[DATA] Sampled {sample_size} rows from dataset before balancing")

    # Build conditional probability matrix and perform NNLS sampling
    cooc_norm = build_conditional_prob_matrix(data, labels)
    resampled_df = nnls_sample(data, labels, target_count, cooc_norm)
    data = resampled_df

    # Check if required columns exist
    required_columns = ['report'] + labels
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

    # Feature column (text data)
    X = data[['report']]

    # Label columns (multi-label targets)
    y = data[labels]

    # Initialize the stratified shuffle split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

    # Perform the split
    for train_index, test_index in msss.split(X, y):
        X_train = X.iloc[train_index].reset_index(drop=True)
        X_test = X.iloc[test_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)

    # Print label counts to check for class imbalance
    print("Label counts in y_train:")
    print(y_train.sum())
    print("\nLabel counts in y_test:")
    print(y_test.sum())
    return X_train, X_test, y_train, y_test
