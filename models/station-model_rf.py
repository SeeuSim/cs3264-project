import os
import glob
import datetime
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Data Loading and Combination
# -----------------------------
# Match directories like "./data/11-apr/_global.csv", "./data/12-apr/_global.csv", etc.
file_pattern = os.path.join('.', 'data', '*-apr', '_global.csv')
csv_files = glob.glob(file_pattern)

list_df = []
for file in csv_files:
    # Use the parent directory name (e.g., "11-apr") as the date
    parent_dir = os.path.basename(os.path.dirname(file))
    date_str = parent_dir  # e.g., "11-apr"
    
    # Parse the date using a fixed year (e.g., 2025) to compute day information
    try:
        dt = datetime.datetime.strptime(date_str.title() + "-2025", "%d-%b-%Y")
        day_of_week = dt.strftime("%A")  # e.g., "Friday"
    except Exception as e:
        print(f"Error parsing date from folder {parent_dir}: {e}")
        continue
    
    # Read CSV file—assumed to include sin_d, cos_d, sin_t, and cos_t columns among others.
    df_temp = pd.read_csv(file)
    df_temp['date'] = date_str
    df_temp['day_of_week'] = day_of_week
    list_df.append(df_temp)

# Combine all files into one DataFrame
df_all = pd.concat(list_df, ignore_index=True)
print("Combined data shape:", df_all.shape)

# ------------------------------
# Data Preprocessing and Encoding
# ------------------------------
# Encode target (assumes values like 'l', 'm', 'h')
le = LabelEncoder()
df_all['target_encoded'] = le.fit_transform(df_all['target'])

# Preserve original station label for grouping
df_all['station_original'] = df_all['station']

# One-hot encode the cl_* columns (using drop_first to avoid dummy variable trap)
df_encoded = pd.get_dummies(df_all, columns=['cl_1', 'cl_2', 'cl_3', 'cl_4', 'cl_5', 'cl_6'], drop_first=True)

# -------------------------
# Train One Model Per Station (Random Forest)
# -------------------------
station_models = {}
stations = df_all['station_original'].unique()

for station in stations:
    print(f"\n=== Training model for station: {station} ===")
    # Filter data for the current station.
    station_data = df_encoded[df_encoded['station_original'] == station].copy()
    # Drop the raw station column if present.
    if 'station' in station_data.columns:
        station_data.drop(columns=['station'], inplace=True)
    
    # Define feature columns by excluding target, identifiers, and metadata.
    exclude_cols = ['target', 'target_encoded', 'station_original', 'date', 'day_of_week']
    feature_cols = [col for col in station_data.columns if col not in exclude_cols]
    
    X = station_data[feature_cols]
    y = station_data['target_encoded']
    
    # If not at least two classes are present, store the majority class.
    if y.nunique() < 2:
        majority_class = y.iloc[0]
        print(f"  Station {station} has only one class. Using majority prediction: {le.inverse_transform([majority_class])[0]}")
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

    # Split data (80%-20%) with stratification.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train a Random Forest Classifier.
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Cross-validation and reporting.
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
# Prediction Function (Revised for Random Forest)
# -------------------------
def predict_for_station_day_time(day_input, time_input, station_input, tol=1e-5):
    """
    Given a day (e.g., "Friday"), a time (HH:MM format), and a station,
    this function:
      1. Filters the data by station and day (using the day_of_week column).
      2. Converts the time input into its cyclic (sin_t, cos_t) representation.
      3. Tries to find rows where the time matches within a tolerance.
      4. If no exact time match is found, selects the nearest row based on the Euclidean distance (computed solely on the time features).
      5. Uses the station's Random Forest model to predict the target; if no model was trained, returns the majority class.
    """
    # Filter by day_of_week.
    day_input_cap = day_input.strip().capitalize()
    valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if day_input_cap not in valid_days:
        print("Invalid day input. Please enter one of: " + ", ".join(valid_days))
        return None

    # Parse the time and convert to cyclic encoding.
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

    # Filter data for the given station and the specified day.
    df_filtered_day = df_encoded[(df_encoded['station_original'] == station_input_cap) &
                                 (df_encoded['day_of_week'] == day_input_cap)]
    if df_filtered_day.empty:
        print("No data found for the specified station and day.")
        return None

    # Now, filter based on time using the cyclic features sin_t and cos_t.
    mask = (
        (np.abs(df_filtered_day['sin_t'] - sin_t_val) < tol) &
        (np.abs(df_filtered_day['cos_t'] - cos_t_val) < tol)
    )
    data_filtered = df_filtered_day[mask]
    
    # If no rows match within tolerance, select the nearest row based on time cyclic distance.
    if data_filtered.empty:
        print("No exact time match found; selecting nearest row based on time cyclic distance...")
        distances = np.sqrt(
            (df_filtered_day['sin_t'] - sin_t_val)**2 +
            (df_filtered_day['cos_t'] - cos_t_val)**2
        )
        min_idx = distances.idxmin()
        data_filtered = df_filtered_day.loc[[min_idx]]
    
    # Retrieve the model for the station.
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
    # Return a subset of columns for clarity.
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
