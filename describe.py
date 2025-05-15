import pandas as pd
from DSLR.ds_core import describe


if __name__ == "__main__":
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    describe_df = describe(df)
    print(describe_df)