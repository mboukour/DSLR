from DSLR.lr_core import MultiLogisticRegression
from DSLR.dslr_utils import load_model_params, StandardScaler
from DSLR.dslr_math import calculate_mean
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    df.drop(columns=["Hogwarts House", "Birthday", "Best Hand", "First Name", "Last Name"], inplace=True)
    for col in df.columns:
        mean = calculate_mean(df[col])
        df[col] = df[col].fillna(mean)
    df.reset_index(drop=True, inplace=True)
    params = load_model_params()
    means = params[0]
    stds = params[1]
    scaler = StandardScaler(means, stds)
    df = scaler.transform(df)
    params = dict(params[2:])
    model = MultiLogisticRegression(pd.DataFrame([]), pd.Series([]))
    model.load_model(params)
    preds = model.predict(df)
    output = pd.DataFrame(preds, columns=["Predictions"])
    output.to_csv("predictions.csv")