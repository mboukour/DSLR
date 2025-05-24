import pandas as pd
import numpy as np
from DSLR.dslr_math import *
from DSLR.dslr_utils import unique_combinations, next_slot
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
        slot = [0,0]
        fig = plt.figure(figsize=(8, 10))
        gs = gridspec.GridSpec(nrows=5, ncols=3, figure=fig)
        for course in courses:
            ax = fig.add_subplot(gs[self.__slot[0], self.__slot[1]])
            ax.set_title(course)
            for _, house in houses:
                ax.hist(x=house[course], alpha=0.5)
            next_slot(slot, 5, 3)
        plt.tight_layout()


    def show(self) -> None:
        plt.show()

    def savefig(self, save_as: str="histogram.png") -> None:
        plt.savefig(save_as)

class ScatterPlot:
    def __init__(self, df: pd.DataFrame):
        df = df.drop(columns=["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"])
        features = list(df.columns)
        pairs = unique_combinations(features)
        fig = plt.figure(figsize=(70,70))
        gs = gridspec.GridSpec(nrows=13, ncols=6, figure=fig)
        slot = [0,0]
        for course_1, course_2 in pairs:
            ax = fig.add_subplot(gs[slot[0], slot[1]])
            ax.set_title(f"{course_1} vs {course_2}")
            ax.scatter(x=df[course_1], y=df[course_2], color="blue")
            next_slot(slot, 13, 6)
        plt.tight_layout()

    def show(self) -> None:
        plt.show()

    def savefig(self, save_as: str="scatter.png") -> None:
        plt.savefig(save_as)

class PairPlot:
    def __init__(self, df: pd.DataFrame):
        color_map = {'Gryffindor': 'red', 'Slytherin': 'green', 'Ravenclaw': 'blue', 'Hufflepuff': 'yellow'}
        colors = df["Hogwarts House"].map(color_map)
        df = df.drop(columns=["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"])
        features = list(df.columns)
        fig = plt.figure(figsize=(100,100))
        gs = gridspec.GridSpec(nrows=13, ncols=13, figure=fig)
        slot = [0,0]
        n = len(features)

        for i in range(n):
            for j in range(n):
                ax = fig.add_subplot(gs[slot[0], slot[1]])
                ax.set_xticks([])
                ax.set_yticks([])
                if i == j:
                    ax.hist(x=df[features[i]], color="grey")
                else:
                    ax.scatter(x=df[features[i]], y=df[features[j]], c=colors)
                if i == n - 1:
                    ax.set_xlabel(features[j], fontsize=40)
                if j == 0:
                    ax.set_ylabel(features[i], fontsize=40)
                next_slot(slot, 13, 13)
        plt.tight_layout()

    def show(self) -> None:
        plt.show()

    def savefig(self, save_as:str = "pair_plot.png") -> None:
        plt.savefig(save_as)
