from datetime import datetime
from os import path

import joblib

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier


RANDOM_STATE = 42


def read_data(paths: list[str]):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        dfs.append(df)

    df_glob = pd.concat(dfs, ignore_index=True)

    return df_glob


def train_model(paths: list[str]):
    df_glob = read_data(paths)

    le = LabelEncoder()
    df_glob["target_enc"] = le.fit_transform(df_glob["target"])

    df_glob = pd.get_dummies(
        df_glob,
        columns=["cl_1", "cl_2", "cl_3", "cl_4", "cl_5", "cl_6"],
        drop_first=True,
    )

    excl = ["target", "target_enc", "station"]
    features = [col for col in df_glob.columns if col not in excl]

    X = df_glob[features]
    y = df_glob["target_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=False
    )

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"  Mean CV Accuracy: {cv_scores.mean():.2f}")

    y_pred = model.predict(X_test)
    labels_present = sorted(set(y_test))
    target_names_present = le.inverse_transform(labels_present)
    print("  Classification Report:")
    print(
        classification_report(
            y_test, y_pred, labels=labels_present, target_names=target_names_present
        )
    )

    joblib.dump(model, f'glob-gb-{datetime.now().strftime('%d-%m-%YT%H:%M:%S')}.joblib')


if __name__ == "__main__":
    days = list(map(lambda d: f"{d}-apr", list(range(11, 13 + 1))))

    data_paths = [path.abspath(path.join("..", "data", d, "_global.csv")) for d in days]

    train_model(data_paths)
    # print(pd.read_csv(data_paths[0], index_col=0).head())
