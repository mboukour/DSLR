import pandas as pd
import numpy as np
from DSLR.dslr_math import *



def describe(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=["number"]).columns
    indexes = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    data = [
        [calculate_count(df[col]) for col in numerical_cols],
        [calculate_mean(df[col]) for col in numerical_cols],
        [calculate_std(df[col]) for col in numerical_cols],
        [calculate_min(df[col]) for col in numerical_cols],
        [calculate_quantile(df[col], 0.25) for col in numerical_cols],
        [calculate_median(df[col]) for col in numerical_cols],
        [calculate_quantile(df[col], 0.75) for col in numerical_cols],
        [calculate_max(df[col]) for col in numerical_cols]
    ]
    describe_df = pd.DataFrame(data=data, index=indexes, columns=numerical_cols)
    describe_df.map(lambda x: f"{x:.2f}")
    return describe_df


