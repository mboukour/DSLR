import pandas as pd
import matplotlib.pyplot as plt
from DSLR.ds_core import PairPlot

if __name__ == "__main__":
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    pair = PairPlot(df)
    pair.savefig("images/pair_plot.png")