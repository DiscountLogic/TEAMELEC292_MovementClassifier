import h5py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

hdf5_file = "accelerometer_data.h5"

def load_activity_data(person, activity):
    with h5py.File(hdf5_file, "r") as hdf:
        preprocessed_data = hdf[f"preprocessed_data/{person}/{activity}"]
        all_data = [np.array(preprocessed_data[speed][position])
                    for speed in preprocessed_data
                    for position in preprocessed_data[speed]]
    return pd.DataFrame(np.vstack(all_data), columns=["Time (s)", "Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"])

def extract_features(window):
    features = {}
    for i, axis in enumerate(["Accel_X", "Accel_Y", "Accel_Z"]):
        accel = window[:, i]
        features[f"max_{axis}"] = np.max(accel)
        features[f"min_{axis}"] = np.min(accel)
        features[f"range_{axis}"] = np.ptp(accel)
        features[f"mean_{axis}"] = np.mean(accel)
        features[f"std_{axis}"] = np.std(accel)
        features[f"rms_{axis}"] = np.sqrt(np.mean(accel ** 2))
        features[f"skew_{axis}"] = skew(accel)
        features[f"kurtosis_{axis}"] = kurtosis(accel)
        fft_vals = np.abs(fft(accel))
        features[f"fft_peak_{axis}"] = np.argmax(fft_vals[1:])
        features[f"fft_mean_{axis}"] = np.mean(fft_vals)
    return features

def normalize_features(df):
    scaler = StandardScaler()
    cols = [col for col in df.columns if col not in ['Activity', 'Person']]
    df[cols] = scaler.fit_transform(df[cols])
    return df

def process_all_people():
    people = ["Alisa", "Brian", "Cissi"]
    activities = ["Walking", "Jumping"]
    all_data = []

    for person in people:
        for activity in activities:
            df = load_activity_data(person, activity)
            window_size = 100  # 5s window @100Hz
            windows = [df.iloc[i:i+window_size, 1:4].values
                       for i in range(0, len(df), window_size)
                       if len(df.iloc[i:i+window_size]) == window_size]
            features = [extract_features(w) for w in windows]
            df_features = pd.DataFrame(features)
            df_features["Activity"] = activity
            df_features["Person"] = person
            all_data.append(df_features)

    return normalize_features(pd.concat(all_data, ignore_index=True))

def save_segmented_data_to_hdf5(df):
    X = df.drop(columns=["Activity", "Person"]).to_numpy()
    y = df["Activity"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    with h5py.File(hdf5_file, "a") as hdf:
        if "Segmented data" in hdf:
            del hdf["Segmented data"]
        seg_group = hdf.create_group("Segmented data")
        train_group = seg_group.create_group("Train")
        test_group = seg_group.create_group("Test")
        train_group.create_dataset("features", data=X_train)
        train_group.create_dataset("labels", data=y_train.astype("S"))
        test_group.create_dataset("features", data=X_test)
        test_group.create_dataset("labels", data=y_test.astype("S"))

df_features_normalized = process_all_people()
save_segmented_data_to_hdf5(df_features_normalized)
