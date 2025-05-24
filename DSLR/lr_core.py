import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log


class LogisticRegression:
    def __init__(self, train: str, X: np.array=None, y: np.array=None, weights:np.array=None):
        self.__train = train
        if  weights:
            self.__weights = weights
            return
        self.__X = np.column_stack((X, np.ones(X.shape[0])))
        self.__n = len(self.__X)
        self.__y = y
        self.__weights = np.zeros(self.__X.shape[1])
        self.__learning_rate = 0.1
        self.__max_epoch = 400
        self.__n = len(self.__X)
        self.__log_loss_history = []

    @staticmethod
    def __sigmoid(v):
        return 1 / (1 + np.exp(-v))


    def train(self):
        for _ in range(self.__max_epoch):
            logits = np.dot(self.__X, self.__weights)
            predictions = self.__sigmoid(logits)
            gradient = 1 / self.__n * np.dot(self.__X.T, (self.__y - predictions))
            self.__weights += self.__learning_rate * gradient
            error = self.__log_loss(predictions)
            self.__log_loss_history.append(error)
    

    def predict(self, X: np.array):
        logits = np.dot(X, self.__weights)
        probs = self.__sigmoid(logits)
        return probs

    def save_log_loss(self, save_to:str = None):
        if not save_to:
            save_to = f"{self.__train}_lr.png"
        plt.scatter(range(self.__max_epoch), self.__log_loss_history)
        plt.xlabel("Epochs")
        plt.xlabel("Log Loss")
        plt.title(f"{self.__train}: Log loss per epochs")
        plt.savefig(save_to)

    def __log_loss(self, preds: np.array) -> float:
        epsilon = 1e-15
        total_loss = 0
        for y, p in zip(self.__y, preds):
            p = min(max(p, epsilon), 1 - epsilon) # p can be between (epsilon) and (1 - epsilon)
            loss = y * log(p) + (1 - y) * log(1 - p)
            total_loss += loss
        return - total_loss / self.__n  # negative avg loss


    def get_final_params(self) -> list:
        return list(self.__weights)

class MultiLogisticRegression:
    def __init__(self, X: pd.DataFrame | None, y: pd.Series | None):
        self.__models = {}
        self.__load_mode = True
        if X.empty or y.empty:
            return
        self.__load_mode = False
        y_vc = y.value_counts()
        for i in range(len(y_vc)):
            current_house = y_vc.index[i]
            new_y = y.map(lambda house: 1 if house == current_house else 0)
            current_model = LogisticRegression(current_house, X, new_y)
            self.__models[current_house] = current_model

    def train(self) -> None:
        if self.__load_mode:
            raise RuntimeError("Load model mode, can't train")
        for model in self.__models.values():
            model.train()

    def load_model(self, params:dict):
        for name, weights in params.items():
            self.__models[name] = LogisticRegression(name, weights=weights)

    def predict(self, X: pd.DataFrame) -> list:
        if self.__load_mode and len(self.__models) == 0:
            raise RuntimeError("Load model before predicting")
        X = np.column_stack((X.to_numpy(), np.ones(X.shape[0])))
        probas = [model.predict(X) for model in self.__models.values()]
        probas = np.stack(probas, axis=1)
        preds = np.array([])
        for proba in probas:
            preds = np.append(preds, np.argmax(proba))
        preds = preds.astype(int)
        keys = list(self.__models.keys())
        res = [keys[i] for i in preds]
        return res
    
    def get_final_params(self) -> dict:
        res = []
        for name, model in self.__models.items():
            res.append((name, model.get_final_params())) 
        return res
    


    
