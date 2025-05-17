import pandas as pd
import numpy as np
from DSLR.dslr_math import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
        courses = df.columns[1:]
        houses = df.groupby("Hogwarts House")[courses] #group and remove House Name
        self.__slot = [0,0]
        self.__cols = 3
        self.__rows = 5
        fig = plt.figure(figsize=(8, 10))
        gs = gridspec.GridSpec(nrows=5, ncols=3, figure=fig)
        for course in courses:
            ax = fig.add_subplot(gs[self.__slot[0], self.__slot[1]])
            ax.set_title(course)
            for name, house in houses:
                ax.hist(x=house[course], alpha=0.5)
            self.__next_slot()
        plt.tight_layout()
        
            
    def __next_slot(self) -> None:
        if self.__slot[1] < self.__cols - 1:
            self.__slot[1] += 1
        elif self.__slot[0] < self.__rows - 1:
            self.__slot[0] += 1
            self.__slot[1] = 0



    def show(self) -> None:
        plt.show()

    def savefig(self, save_as: str="histogram.png") -> None:
        plt.savefig(save_as)
