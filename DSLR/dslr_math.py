import numpy as np
import pandas as pd
from math import sqrt

def calculate_mean(data: np.array) -> float:
    data = data[~np.isnan(data)]
    return (sum(data) / len(data))


def calculate_std(data: np.array) -> float:
    data = data[~np.isnan(data)]
    mean = calculate_mean(data)
    squared_diff = (data - mean) ** 2
    return sqrt(sum(squared_diff) / len(data))


def calculate_count(data: pd.Series) -> float:
    data = ~np.isnan(data)
    return sum(data)

def calculate_min(data: np.array) -> float:
    data = data[~np.isnan(data)]
    current_min = float('inf')
    for x in data:
        current_min = x if x < current_min else current_min
    return current_min

def calculate_max(data: np.array) -> float:
    data = data[~np.isnan(data)]
    current_max = float('-inf')
    for x in data:
        current_max = x if x > current_max else current_max
    return current_max


def calculate_quantile(data: np.array, percentile: float) -> float:
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
    data = data[~np.isnan(data)]
    data = sorted(data)
    size = int(len(data))
    if size % 2 != 0:
        return data[(size) // 2]
    return (data[size // 2 - 1] + data[size // 2]) / 2