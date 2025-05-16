import pandas as pd
from DSLR.ds_core import describe
from sys import argv, stderr

if __name__ == "__main__":
    if len(argv) > 2:
        print("Usage: python3 describe.py {path/to/dataset}", file=stderr)
        exit(1)
    try:
        df = pd.read_csv(argv[1], index_col="Index")
        describe_df = describe(df)
        print(describe_df)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)