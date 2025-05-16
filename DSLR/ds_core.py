import pandas as pd
import numpy as np
from DSLR.dslr_math import *
import matplotlib.pyplot as plt


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


class Histogram:
    def __init__(self, df: pd.DataFrame):
        df = df.drop(columns=["First Name", "Last Name", "Birthday", "Best Hand"])
        cph = df.groupby("Hogwarts House")[df.columns[1:]] #group and remove House Name
        data = []
        indexes = cph.groups.keys()
        for name, group in cph:
            mean_list = []
            for course in group.columns:
                mean = calculate_mean(group[course])
                mean_list.append(mean)
            data.append(mean_list)
        course_means_df = pd.DataFrame(data=data, index=indexes, columns=df.columns[1:])
        stds = {}
        for course in course_means_df.columns:
            stds[course] = calculate_std(df[course])
        stds = dict(sorted(stds.items(), key=lambda x: x[1]))
        homogeneous_course = next(iter(stds))
        hc_groups = df.groupby("Hogwarts House")[homogeneous_course]
        for name, group in hc_groups:
            plt.hist(group, alpha=0.5, label=name)
        plt.title(f"Histogram representing {homogeneous_course} scores acorss all houses")
        plt.xlabel("Scores")
        plt.legend()

    def show(self) -> None:
        plt.show()

    def savefig(self, save_as: str="histogram.png") -> None:
        plt.savefig(save_as)
