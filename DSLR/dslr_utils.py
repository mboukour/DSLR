import pandas as pd
import numpy as np
from DSLR.dslr_math import calculate_mean, calculate_std, calculate_quantile
import json 

def unique_combinations(data: list) -> list:
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
    if slot[1] < ncols - 1:
        slot[1] += 1
    elif slot[0] < nrows - 1:
        slot[0] += 1
        slot[1] = 0


class StandardScaler:
    def __init__(self, means:list = [], stds:list = []):
        self.__means = means
        self.__stds = stds
    

    def fit(self, X: pd.DataFrame) -> None:
        self.__means = []
        self.__stds = []
        for column in X.columns:
            self.__means.append(calculate_mean(X[column]))
            self.__stds.append(calculate_std(X[column]))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for i, column in zip(range(len(X.columns)), X.columns):
            X[column] = (X[column] - self.__means[i]) / self.__stds[i]
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
    
    def get_scaling_params(self) -> tuple:
        return self.__means, self.__stds
    

def test_train_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42) -> tuple:
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")
    # shuffle indexes
    shuffled_indexes = X.sample(frac=1, random_state=random_state).reset_index(drop=True).index

    X = X.loc[shuffled_indexes].reset_index(drop=True)
    y = y.loc[shuffled_indexes].reset_index(drop=True)
    split_index = int(len(X) * (1 - test_size)) # operation may return flaot

    #slice from start to split_index
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    #slice from split_index to end
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def accuracy(y: list, pred: list):
    if len(y) != len(pred):
        raise ValueError("The arrays must be same shape")
    correct = [Yi == Pi for Yi, Pi in zip(y, pred)]
    return sum(correct) / len(y)


def load_model_params(file:str = "model.json"):
    with open(file, 'r') as f:
        data = json.load(f)
        return data