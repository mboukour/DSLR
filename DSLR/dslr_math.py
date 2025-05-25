import numpy as np
import pandas as pd
from math import sqrt

def calculate_mean(data: np.array) -> float:
    """
    Calculate the mean (average) of the given numeric data, ignoring NaN values.

    Args:
        data (np.array): Input array containing numeric data, may include NaNs.

    Returns:
        float: Mean of the non-NaN values in the data.
    """
    data = data[~np.isnan(data)]
    return (sum(data) / len(data))


def calculate_std(data: np.array) -> float:
    """
    Calculate the standard deviation of the given numeric data, ignoring NaN values.

    Args:
        data (np.array): Input array containing numeric data, may include NaNs.

    Returns:
        float: Standard deviation of the non-NaN values in the data.
    """
    data = data[~np.isnan(data)]
    mean = calculate_mean(data)
    squared_diff = (data - mean) ** 2
    return sqrt(sum(squared_diff) / len(data))


def calculate_count(data: pd.Series) -> float:
    """
    Count the number of non-NaN values in the pandas Series.

    Args:
        data (pd.Series): Input pandas Series which may contain NaNs.

    Returns:
        float: Count of non-NaN values in the series.
    """
    data = ~np.isnan(data)
    return sum(data)


def calculate_min(data: np.array) -> float:
    """
    Calculate the minimum value from the numeric data, ignoring NaN values.

    Args:
        data (np.array): Input array containing numeric data, may include NaNs.

    Returns:
        float: Minimum value among the non-NaN values.
    """
    data = data[~np.isnan(data)]
    current_min = float('inf')
    for x in data:
        current_min = x if x < current_min else current_min
    return current_min


def calculate_max(data: np.array) -> float:
    """
    Calculate the maximum value from the numeric data, ignoring NaN values.

    Args:
        data (np.array): Input array containing numeric data, may include NaNs.

    Returns:
        float: Maximum value among the non-NaN values.
    """
    data = data[~np.isnan(data)]
    current_max = float('-inf')
    for x in data:
        current_max = x if x > current_max else current_max
    return current_max


def calculate_quantile(data: np.array, percentile: float) -> float:
    """
    Calculate the quantile value at the specified percentile of the data, ignoring NaNs.

    Args:
        data (np.array): Input array containing numeric data, may include NaNs.
        percentile (float): Desired percentile between 0 and 1 inclusive.

    Returns:
        float: Value at the specified percentile.

    Raises:
        ValueError: If percentile is not between 0 and 1.
    """
    data = data[~np.isnan(data)]    
    if percentile < 0 or percentile > 1:
        raise ValueError("percentile should be between 0 and 1")

    if percentile == 0.5:
        return calculate_median(data)

    data = sorted(data)
    if percentile == 0:
        return data[0]
    elif percentile == 1:
        return data[-1]
    rank = percentile * (len(data) - 1)
    rank_int = int(rank)
    fraction = rank - rank_int
    value = data[rank_int] + fraction * (data[rank_int + 1] - data[rank_int])
    return value


def calculate_median(data: np.array):
    """
    Calculate the median (50th percentile) of the numeric data, ignoring NaN values.

    Args:
        data (np.array): Input array containing numeric data, may include NaNs.

    Returns:
        float: Median value of the non-NaN data.
    """
    data = data[~np.isnan(data)]
    data = sorted(data)
    size = int(len(data))
    if size % 2 != 0:
        return data[(size) // 2]
    return (data[size // 2 - 1] + data[size // 2]) / 2
