import pandas as pd
from DSLR.ds_core import Histogram


if __name__  == "__main__":
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    hist = Histogram(df)
    hist.savefig("images/histogram.png")
