

from DSLR.dslr_math import calculate_mean, calculate_std
import pandas as pd



if __name__  == "__main__":
    df = pd.read_csv("datasets/dataset_train.csv", index_col="Index")
    df.drop(columns=["First Name", "Last Name", "Birthday", "Best Hand"], inplace=True)
    houses = df.groupby("Hogwarts House")
    courses = df.columns[1:]
    data = []
    for name, group in houses:
        mpc = []
        for course in courses:
            mean = calculate_mean(group[course])
            mpc.append(mean)
        data.append(mpc)