import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log


class LogisticRegression:
    def __init__(self, class_name: str, X: pd.DataFrame=None, y: np.array=None, weights:np.array=None, mode:str="batch"):
        self.__class_name = class_name
        if mode not in ["batch", "mini-batch", "stochastic"]:
            raise ValueError("Invalid training mode")
        self.__mode = mode
        if  weights:
            self.__weights = weights
            return
        self.__X = X
        self.__X["bias"] = 1
        self.__n = len(self.__X)
        self.__y = y
        self.__weights = np.zeros(self.__X.shape[1])
        self.__learning_rate = 0.1
        self.__max_epoch = 400
        self.__n = len(self.__X)
        self.__log_loss_history = []
        self.__stop_gradient_norm = 0.01

    @staticmethod
    def __sigmoid(v):
        return 1 / (1 + np.exp(-v))

    def __bactch_gd(self):
        for _ in range(self.__max_epoch):
            logits = np.dot(self.__X, self.__weights)
            predictions = self.__sigmoid(logits)
            gradient = 1 / self.__n * np.dot(self.__X.T, (self.__y - predictions))
            if np.linalg.norm(gradient) < self.__stop_gradient_norm:
                return
            self.__weights += self.__learning_rate * gradient
            error = self.__log_loss(predictions, self.__n)
            self.__log_loss_history.append(error)

    def __mini_batch_gd(self):
        batch_size = 32
        for _ in range(self.__max_epoch):
            indices = np.random.permutation(len(self.__X))
            shuffled_x = self.__X.iloc[indices].reset_index(drop=True).to_numpy()
            shuffled_y = self.__y.iloc[indices].reset_index(drop=True).to_numpy()
            x_batches = np.array_split(shuffled_x, len(shuffled_x) // batch_size)
            y_batches = np.array_split(shuffled_y, len(shuffled_y) // batch_size)
            stop_training = False
            loss = 0
            visited_rows = 0
            for i, batch in enumerate(x_batches):
                logits = np.dot(batch, self.__weights)
                predictions = self.__sigmoid(logits)
                gradient = 1 / len(batch) * np.dot(batch.T, (y_batches[i] - predictions))
                if np.linalg.norm(gradient) < self.__stop_gradient_norm:
                    stop_training = True
                    break
                self.__weights += self.__learning_rate * gradient
                loss += self.__mini_batch_log_loss(predictions, y_batches[i])
                visited_rows += len(batch)
            self.__log_loss_history.append(loss / visited_rows)
            if stop_training:
                return


    def __stochastic_gd(self):
        for _ in range(self.__max_epoch):
            indices = np.random.permutation(len(self.__X))
            shuffled_x = self.__X.iloc[indices].reset_index(drop=True)
            shuffled_y = self.__y.iloc[indices].reset_index(drop=True)
            loss = 0
            stop_training = False
            for i,row in enumerate(shuffled_x.itertuples(index=False)):
                sample = np.array(row)
                current_y = shuffled_y.iloc[i]
                logit = np.dot(sample, self.__weights)
                prediction = self.__sigmoid(logit)
                gradient = sample * (current_y - prediction)
                if np.linalg.norm(gradient) < self.__stop_gradient_norm:
                    stop_training = True
                    break
                loss += self.__stochastic_log_loss(prediction, current_y)
                self.__weights += self.__learning_rate * gradient
            self.__log_loss_history.append(loss / self.__n)
            if stop_training:
                return


    def fit(self):
        if self.__mode == "batch":
            self.__bactch_gd()
        elif self.__mode == "stochastic":
            self.__stochastic_gd()
        elif self.__mode == "mini-batch":
            self.__mini_batch_gd()


    def predict(self, X: np.array):
        logits = np.dot(X, self.__weights)
        probs = self.__sigmoid(logits)
        return probs

    def scatter_log_loss(self):
        plt.scatter(list(range(len(self.__log_loss_history))), self.__log_loss_history, s=10, label=self.__class_name)


    def __stochastic_log_loss(self, pred: float, target: float) -> float:
        epsilon = 1e-15
        pred = min(max(pred, epsilon), 1 - epsilon)
        loss = target * log(pred) + (1 - target) * log(1 - pred)
        return -loss

    def __log_loss(self, preds: np.array) -> float:
        epsilon = 1e-15
        total_loss = 0
        for y, p in zip(self.__y, preds):
            p = min(max(p, epsilon), 1 - epsilon) # p can be between (epsilon) and (1 - epsilon)
            loss = y * log(p) + (1 - y) * log(1 - p)
            total_loss += loss
        return - total_loss / self.__n  # negative avg loss

    def __mini_batch_log_loss(self, preds: np.array, targets: np.array) -> float:
        epsilon = 1e-15
        total_loss = 0
        for y, p in zip(targets, preds):
            p = min(max(p, epsilon), 1 - epsilon) # p can be between (epsilon) and (1 - epsilon)
            loss = y * log(p) + (1 - y) * log(1 - p)
            total_loss += loss
        return - total_loss   # negative avg loss

    def get_model_params(self) -> list:
        return list(self.__weights)

class MultiLogisticRegression:
    def __init__(self, X: pd.DataFrame | None, y: pd.Series | None, mode:str="batch"):
        self.__models = {}
        self.__load_mode = True
        if X.empty or y.empty:
            return
        self.__load_mode = False
        y_vc = y.value_counts()
        for i in range(len(y_vc)):
            current_house = y_vc.index[i]
            new_y = y.map(lambda house: 1 if house == current_house else 0)
            current_model = LogisticRegression(current_house, X, new_y, mode=mode)
            self.__models[current_house] = current_model

    def fit(self) -> None:
        if self.__load_mode:
            raise RuntimeError("Load model mode, can't train")
        for model in self.__models.values():
            model.fit()

    def load_model(self, params:dict):
        if not self.__load_mode:
            raise RuntimeError("Can't load model in training mode")
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

    def get_model_params(self) -> dict:
        res = []
        for name, model in self.__models.items():
            res.append((name, model.get_model_params())) 
        return res

    def scatter_log_loss(self, save_to:str="images/logloss.png") -> None:
        for model in self.__models.values():
            model.scatter_log_loss()
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Log Loss")
        plt.title("Log Loss over Epochs")
        plt.savefig(save_to)
