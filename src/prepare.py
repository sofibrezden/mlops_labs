import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split


def preprocess(df):

    df["datetime"] = pd.to_datetime(df["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    df = df.drop(["datetime", "casual", "registered"], axis=1)

    return df


if __name__ == "__main__":

    input_file = sys.argv[1]      # data/raw/dataset.csv
    output_dir = sys.argv[2]      # data/prepared

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file)

    df = preprocess(df)
    df = df.sort_values("day")

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Data prepared successfully.")