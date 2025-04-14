import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import re
import pathlib

"""
FLAGS
"""
raw_columns = ["station", "start_time", "end_time", "crowd_level"]

#######################################################################
"""
FUNCTIONS
"""


def get_relevant_cols(df: pd.DataFrame):
    df["start_time"] = pd.to_datetime(
        df["start_time"].str.replace(r"\+08:00$", "", regex=True)
    )
    return df[raw_columns]


def clean_crowd_levels(df: pd.DataFrame):
    df["crowd_level"] = df["crowd_level"].replace("na", "l")
    return df


"""
Sort by BP1, BP2, ... (numerical) as opposed to BP1, BP10, ... (alphabetical)
"""


def get_sorted_s_code(s: str):
    return re.sub(r"\d+$", lambda m: m.group().zfill(2), s)


def get_station_codes_sorted(df: pd.DataFrame):
    return sorted(list(df["station"].unique()), key=get_sorted_s_code)


def get_folder_name(in_csv: str):
    fol = re.search(r"([\-\d\w]+)", in_csv)
    if fol is not None:
        return fol.group()
    return ""


def create_features(in_csv: str):
    out_folder_name = get_folder_name(in_csv)
    out_folder_path = pathlib.Path(out_folder_name)
    if not out_folder_path.is_dir():
        out_folder_path.mkdir()

    df = pd.read_csv(in_csv)
    df = get_relevant_cols(df)
    df = clean_crowd_levels(df)

    station_codes = get_station_codes_sorted(df)

    global_features = []

    expected_delta = pd.Timedelta("10min")
    mins_in_day = 24 * 60

    for station in station_codes:
        features = []
        station_frame = df[df["station"] == station]
        for i in range(0, len(station_frame) - 7):
            rows = station_frame[i : i + 7]
            deltas = rows["start_time"].diff().dropna()

            # Not all 10mins
            if not all(d == expected_delta for d in deltas):
                continue

            first_row = rows.iloc[0]

            # Time Features
            min_of_day = (
                first_row["start_time"].hour * 60 + first_row["start_time"].minute
            )
            sin_t = np.sin(2 * np.pi * min_of_day / mins_in_day)
            cos_t = np.cos(2 * np.pi * min_of_day / mins_in_day)

            # Day features
            day_of_week = first_row["start_time"].day_of_week
            sin_d = np.sin(2 * np.pi * day_of_week / 7)
            cos_d = np.cos(2 * np.pi * day_of_week / 7)

            # Crowd Level windows
            cl_features = rows.iloc[:6]["crowd_level"].values
            cl_dict = {f"cl_{j+1}": cl_features[j] for j in range(6)}

            r = {
                "sin_d": sin_d,
                "cos_d": cos_d,
                "sin_t": sin_t,
                "cos_t": cos_t,
                **cl_dict,
                "target": rows.iloc[6]["crowd_level"],
            }

            # Final row
            features.append(r)
            global_features.append({"station": station, **r})

        features = pd.DataFrame(features)
        features.to_csv(out_folder_name + f"/{station}.csv")

    # One Hot Encode station code
    encoder = OneHotEncoder(sparse_output=False, categories=[station_codes])
    global_features = pd.DataFrame(global_features)
    stations = global_features["station"].values.reshape(-1, 1)
    station_encoded = encoder.fit_transform(stations)
    col_names = encoder.get_feature_names_out(["station"])
    station_encoded = pd.DataFrame(
        station_encoded, columns=col_names, index=global_features.index
    )
    global_features = pd.concat([global_features, station_encoded], axis=1)

    global_features.to_csv(out_folder_name + "/_global.csv")


if __name__ == "__main__":
    for data in [
        # "11-apr.csv",
        # "12-apr.csv",
        "13-apr.csv"
    ]:
        create_features(data)
