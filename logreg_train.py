import pandas as pd
from DSLR.dslr_utils import StandardScaler, test_train_split, accuracy
from DSLR.lr_core import MultiLogisticRegression
from DSLR.dslr_math import calculate_mean
import json
import time

if __name__  == "__main__":
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    df.drop(columns=["Birthday", "Best Hand", "First Name", "Last Name"], inplace=True)
    y = df["Hogwarts House"]
    df.drop(columns=["Hogwarts House"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    for col in df.columns:
        mean = calculate_mean(df[col])
        df[col] =  df[col].fillna(mean)
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    X_train, X_test, y_train, y_test = test_train_split(df, y)
    model = MultiLogisticRegression(X_train, y_train, mode="mini-batch")
    model.fit()
    model.scatter_log_loss()
    predictions = model.predict(X_test)
    print(f"Model trained with {(accuracy(y_test, predictions) * 100):.2f}% accuracy")
    params = list(scaler.get_scaling_params())
    params.extend(model.get_model_params()) 
    with open("model.json", "w") as f:
        json.dump(params, f, indent=4)
