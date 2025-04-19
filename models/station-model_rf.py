import os
import glob
import datetime
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Data Loading and Combination
file_pattern = os.path.join('..', 'data', '*-apr', '_global.csv')
csv_files = glob.glob(file_pattern)

list_df = []
for file in csv_files:
    parent_dir = os.path.basename(os.path.dirname(file))
    date_str = parent_dir 
    try:
        dt = datetime.datetime.strptime(date_str.title() + "-2025", "%d-%b-%Y")
        day_of_week = dt.strftime("%A")
    except Exception as e:
        print(f"Error parsing date from folder {parent_dir}: {e}")
        continue
    df_temp = pd.read_csv(file)
    df_temp['date'] = date_str
    df_temp['day_of_week'] = day_of_week
    list_df.append(df_temp)

# Combine all files into one DataFrame.
df_all = pd.concat(list_df, ignore_index=True)

# Data Preprocessing and Encoding
le = LabelEncoder()
df_all['target_encoded'] = le.fit_transform(df_all['target'])

df_all['station_original'] = df_all['station']

df_encoded = pd.get_dummies(df_all, columns=['cl_1','cl_2','cl_3','cl_4','cl_5','cl_6'], drop_first=True)

# Train/Test Split and Model Training per Station
station_predictions = {}

stations = df_all['station_original'].unique()
for station in stations:
    print(f"\n=== Processing station: {station} ===")

    station_data = df_encoded[df_encoded['station_original'] == station].copy()
    if 'station' in station_data.columns:
        station_data.drop(columns=['station'], inplace=True)
    
    exclude_cols = ['target','target_encoded','station_original','date','day_of_week']
    feature_cols = [col for col in station_data.columns if col not in exclude_cols]
    
    X = station_data[feature_cols]
    y = station_data['target_encoded']
    
    if y.nunique() < 2:
        majority_class = y.iloc[0]
        print(f"Station {station} has only one class. Using majority prediction: {le.inverse_transform([majority_class])[0]}")
        station_predictions[station.upper()] = {
            'true': y,
            'pred': np.repeat(majority_class, len(y))
        }
        continue
    
    label_counts = Counter(y)
    if min(label_counts.values()) < 2:
        majority_class = y.mode()[0]
        print(f"Station {station} has a class with less than 2 samples. Using majority prediction: {le.inverse_transform([majority_class])[0]}")
        station_predictions[station.upper()] = {
            'true': y,
            'pred': np.repeat(majority_class, len(y))
        }
        continue
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    station_predictions[station.upper()] = {
        'true': y_test.values,
        'pred': predictions
    }
    
    print(f"Classification Report for station {station}:")
    all_classes = np.unique(np.concatenate([y_test, predictions]))
    print(classification_report(
        y_test, 
        predictions, 
        labels=all_classes, 
        target_names=le.inverse_transform(all_classes)
    ))
    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, predictions)*100))

# Overall Evaluation Across Stations
all_true = np.concatenate([val['true'] for val in station_predictions.values()])
all_pred = np.concatenate([val['pred'] for val in station_predictions.values()])
print("\nOverall Classification Report:")
print(classification_report(all_true, all_pred, target_names=le.inverse_transform(sorted(list(set(all_true))))))
print("Overall Accuracy: {:.2f}%".format(accuracy_score(all_true, all_pred)*100))
