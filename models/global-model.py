from os import path

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def read_data(path: str):
    df = pd.read_csv(path)
    return df


def train_model(path: str):
    df = read_data(path)


if __name__ == "__main__":
    days = list(map(lambda d: f"{d}-apr", [11, 12]))
    
    data_paths = [path.abspath(path.join("..", "data", d, "_global.csv")) for d in days]

    for path in data_paths:
        train_model(path)
