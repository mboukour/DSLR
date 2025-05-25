import pandas as pd
import numpy as np
from DSLR.dslr_math import calculate_mean, calculate_std, calculate_quantile
import json 

def unique_combinations(data: list) -> list:
    """
    Generate all unique pairwise combinations (tuples) from the given list without repetition.

    Args:
        data (list): Input list of elements.

    Returns:
        list: List of unique tuple pairs (item1, item2) where item1 comes before item2 in the list.
    """
    combinations = []
    remaining = data[1:]
    for item in data:
        for r in remaining:
            combinations.append((item, r))
        if len(remaining) > 1:
            remaining.pop(0)
        else:
            break
    return combinations

def next_slot(slot: list, nrows: int, ncols: int) -> None:
    """
    Update the given 2D slot position in a grid, moving horizontally first,
    then vertically to the next row when the end of a row is reached.

    Args:
        slot (list): A list with two integers [row, col] representing current position.
        nrows (int): Total number of rows in the grid.
        ncols (int): Total number of columns in the grid.

    Returns:
        None: The slot list is modified in-place.
    """
    if slot[1] < ncols - 1:
        slot[1] += 1
    elif slot[0] < nrows - 1:
        slot[0] += 1
        slot[1] = 0


class StandardScaler:
    """
    A simple standard scaler for pandas DataFrames that standardizes features by removing
    the mean and scaling to unit variance using training data statistics.

    Attributes:
        __means (list): List of means for each feature.
        __stds (list): List of standard deviations for each feature.
    """

    def __init__(self, means:list = [], stds:list = []):
        """
        Initialize the StandardScaler with optional precomputed means and stds.

        Args:
            means (list, optional): List of means for each feature.
            stds (list, optional): List of standard deviations for each feature.
        """
        self.__means = means
        self.__stds = stds
    

    def fit(self, X: pd.DataFrame) -> None:
        """
        Compute the mean and standard deviation for each feature in the DataFrame.

        Args:
            X (pd.DataFrame): Input features to compute scaling parameters from.

        Returns:
            None
        """
        self.__means = []
        self.__stds = []
        for column in X.columns:
            self.__means.append(calculate_mean(X[column]))
            self.__stds.append(calculate_std(X[column]))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the DataFrame using the stored means and standard deviations.

        Args:
            X (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with standardized features.
        """
        X = X.copy()
        for i, column in zip(range(len(X.columns)), X.columns):
            X[column] = (X[column] - self.__means[i]) / self.__stds[i]
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit to the data, then transform it.

        Args:
            X (pd.DataFrame): DataFrame to fit and transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with standardized features.
        """
        self.fit(X)
        return self.transform(X)
    
    def get_scaling_params(self) -> tuple:
        """
        Get the stored scaling parameters.

        Returns:
            tuple: Tuple containing two lists - means and standard deviations of features.
        """
        return self.__means, self.__stds
    

def test_train_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42) -> tuple:
    """
    Split features and labels into training and testing sets with shuffling.

    Args:
        X (pd.DataFrame): Feature dataframe.
        y (pd.Series): Target labels.
        test_size (float, optional): Proportion of data to use for test set (between 0 and 1).
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) split datasets.

    Raises:
        ValueError: If test_size is not between 0 and 1.
    """
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")
    # shuffle indexes
    shuffled_indexes = X.sample(frac=1, random_state=random_state).reset_index(drop=True).index

    X = X.loc[shuffled_indexes].reset_index(drop=True)
    y = y.loc[shuffled_indexes].reset_index(drop=True)
    split_index = int(len(X) * (1 - test_size)) # operation may return float

    #slice from start to split_index
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    #slice from split_index to end
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def accuracy(y: list, pred: list):
    """
    Calculate the accuracy score between true and predicted labels.

    Args:
        y (list): True labels.
        pred (list): Predicted labels.

    Returns:
        float: Accuracy score as a fraction between 0 and 1.

    Raises:
        ValueError: If y and pred have different lengths.
    """
    if len(y) != len(pred):
        raise ValueError("The arrays must be same shape")
    correct = [Yi == Pi for Yi, Pi in zip(y, pred)]
    return sum(correct) / len(y)


def load_model_params(file:str = "model.json"):
    """
    Load model parameters from a JSON file.

    Args:
        file (str, optional): Path to the JSON file containing model parameters.

    Returns:
        dict or list: Parsed model parameters loaded from the file.
    """
    with open(file, 'r') as f:
        data = json.load(f)
        return data
