import pandas as pd
from DSLR.ds_core import Scatter


if __name__ == "__main__":
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    sc = Scatter(df)
    sc.savefig("images/scatter.png")