import os
import glob
import datetime
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

# -----------------------------
# Data Loading and Combination
# -----------------------------
file_pattern = os.path.join('.', 'data', '*-apr', '_global.csv')
csv_files = glob.glob(file_pattern)

list_df = []
for file in csv_files:
    parent_dir = os.path.basename(os.path.dirname(file))
    date_str = parent_dir  # e.g. "11-apr"
    try:
        dt = datetime.datetime.strptime(date_str.title() + "-2025", "%d-%b-%Y")
        day_of_week = dt.strftime("%A")
    except Exception as e:
        print(f"Error parsing date from folder {parent_dir}: {e}")
        continue
    df_temp = pd.read_csv(file)
    df_temp['date'] = date_str
    df_temp['day_of_week'] = day_of_week  # use for filtering by day
    list_df.append(df_temp)

df_all = pd.concat(list_df, ignore_index=True)
print("Combined data shape:", df_all.shape)

# ------------------------------
# Data Preprocessing and Encoding
# ------------------------------
le = LabelEncoder()
df_all['target_encoded'] = le.fit_transform(df_all['target'])
df_all['station_original'] = df_all['station']

df_encoded = pd.get_dummies(df_all, columns=['cl_1', 'cl_2', 'cl_3', 'cl_4', 'cl_5', 'cl_6'], drop_first=True)

# -------------------------
# Train One Gradient Boosting Model Per Station
# -------------------------
station_models = {}
stations = df_all['station_original'].unique()

for station in stations:
    print(f"\n=== Training model for station: {station} ===")
    station_data = df_encoded[df_encoded['station_original'] == station].copy()
    if 'station' in station_data.columns:
        station_data.drop(columns=['station'], inplace=True)
    exclude_cols = ['target', 'target_encoded', 'station_original', 'date', 'day_of_week']
    feature_cols = [col for col in station_data.columns if col not in exclude_cols]
    X = station_data[feature_cols]
    y = station_data['target_encoded']
    
    if y.nunique() < 2:
        majority_class = y.iloc[0]
        print(f"  Station {station} has only one class. Will use majority prediction: {le.inverse_transform([majority_class])[0]}")
        station_models[station.upper()] = {
            'majority_class': majority_class,
            'label_encoder': le
        }
        continue

    label_counts = Counter(y)
    if min(label_counts.values()) < 2:
        majority_class = y.mode()[0]
        print(f"  Station {station} has a class with less than 2 samples. Using majority prediction: {le.inverse_transform([majority_class])[0]}")
        station_models[station.upper()] = {
            'majority_class': majority_class,
            'label_encoder': le
        }
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"  Mean CV Accuracy: {cv_scores.mean():.2f}")
    y_pred = model.predict(X_test)
    labels_present = sorted(set(y_test))
    target_names_present = le.inverse_transform(labels_present)
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, labels=labels_present, target_names=target_names_present))
    
    station_models[station.upper()] = {
        'model': model,
        'features': feature_cols,
        'label_encoder': le
    }

# -------------------------
# Prediction Function (Revised)
# -------------------------
def predict_for_station_day_time(day_input, time_input, station_input, tol=1e-5):
    """
    Given a day (e.g., "Friday"), a time (in HH:MM format), and a station,
    this function converts the day and time into their cyclic representations (for time only),
    filters the combined dataset for rows matching the day (using day_of_week) and time values (within tolerance),
    and returns predictions using the Gradient Boosting model.
    If no model was trained (insufficient class diversity), returns the majority class.
    If no row matches within tolerance, selects the nearest row based on the time (sin_t and cos_t) distance.
    """
    # Filter by day-of-week directly using the day_of_week column.
    day_input_cap = day_input.strip().capitalize()
    valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if day_input_cap not in valid_days:
        print("Invalid day input. Please enter one of: " + ", ".join(valid_days))
        return None

    # Convert time to cyclic encoding.
    try:
        t_obj = datetime.datetime.strptime(time_input.strip(), "%H:%M")
        total_minutes = t_obj.hour * 60 + t_obj.minute
        fraction_of_day = total_minutes / 1440.0
        sin_t_val = np.sin(2 * np.pi * fraction_of_day)
        cos_t_val = np.cos(2 * np.pi * fraction_of_day)
    except Exception as e:
        print(f"Error parsing time input: {e}")
        return None

    station_input_cap = station_input.upper()

    # First, filter by station and day_of_week.
    df_filtered_day = df_encoded[(df_encoded['station_original'] == station_input_cap) &
                                 (df_encoded['day_of_week'] == day_input_cap)]
    if df_filtered_day.empty:
        print("No data found for the specified station and day.")
        return None

    # Then, try to filter rows where the time matches within a tolerance.
    mask = (
        (np.abs(df_filtered_day['sin_t'] - sin_t_val) < tol) &
        (np.abs(df_filtered_day['cos_t'] - cos_t_val) < tol)
    )
    data_filtered = df_filtered_day[mask]
    
    # If no rows match within the tolerance, select the nearest row based on time distance.
    if data_filtered.empty:
        print("No exact time match found; selecting nearest row based on time cyclic distance...")
        distances = np.sqrt(
            (df_filtered_day['sin_t'] - sin_t_val)**2 +
            (df_filtered_day['cos_t'] - cos_t_val)**2
        )
        min_idx = distances.idxmin()
        data_filtered = df_filtered_day.loc[[min_idx]]
    
    # Retrieve station model.
    station_info = station_models.get(station_input_cap)
    if station_info is None:
        print(f"No model available for station {station_input_cap}.")
        return None

    if 'model' in station_info:
        features = station_info['features']
        model = station_info['model']
        predictions = model.predict(data_filtered[features])
        predicted_labels = station_info['label_encoder'].inverse_transform(predictions)
    else:
        majority_class = station_info['majority_class']
        predicted_labels = np.repeat(station_info['label_encoder'].inverse_transform([majority_class])[0],
                                     data_filtered.shape[0])
    
    data_filtered = data_filtered.copy()
    data_filtered['predicted_target'] = predicted_labels
    return data_filtered[['date', 'station_original', 'day_of_week', 'sin_t', 'cos_t', 'predicted_target']]

# -------------------------
# User Interaction
# -------------------------
input_day = input("Enter the day of the week (e.g., 'Friday'): ").strip()
input_time = input("Enter the time (HH:MM, e.g., '14:30'): ").strip()
input_station = input("Enter the station: ").strip()

predicted_results = predict_for_station_day_time(input_day, input_time, input_station)
if predicted_results is not None:
    print("\nPredicted results:")
    print(predicted_results)
