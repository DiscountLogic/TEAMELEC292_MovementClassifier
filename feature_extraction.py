import h5py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler

# Load data from HDF5 file
hdf5_file = "accelerometer_data.h5"


def load_activity_data(person, activity):
    """Loads accelerometer data for a given person and activity."""
    with h5py.File(hdf5_file, "r") as hdf:
        preprocessed_data = hdf[f"preprocessed_data/{person}/{activity}"]

        all_data = []
        for speed in preprocessed_data:
            for position in preprocessed_data[speed]:
                dataset = np.array(preprocessed_data[speed][position])
                all_data.append(dataset)

    data_combined = np.vstack(all_data)
    return pd.DataFrame(data_combined, columns=["Time (s)", "Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"])


def extract_features(window):
    """Extracts 10 features per axis (X, Y, Z)."""
    features = {}

    for i, axis in enumerate(["Accel_X", "Accel_Y", "Accel_Z"]):
        accel = window[:, i]

        # Time-domain features
        features[f"max_{axis}"] = np.max(accel)
        features[f"min_{axis}"] = np.min(accel)
        features[f"range_{axis}"] = np.ptp(accel)
        features[f"mean_{axis}"] = np.mean(accel)
        features[f"std_{axis}"] = np.std(accel)
        features[f"rms_{axis}"] = np.sqrt(np.mean(accel ** 2))  # Root Mean Square
        features[f"skew_{axis}"] = skew(accel)  # Skewness
        features[f"kurtosis_{axis}"] = kurtosis(accel)  # Kurtosis

        # Frequency-domain feature (FFT Peak Frequency and Mean)
        fft_values = np.abs(fft(accel))
        features[f"fft_peak_{axis}"] = np.argmax(fft_values[1:])  # Ignore DC component
        features[f"fft_mean_{axis}"] = np.mean(fft_values)  # Mean frequency amplitude

    return features


def normalize_features(df_features):
    """Applies Z-score standardization (normalization) to all features."""
    scaler = StandardScaler()
    feature_columns = [col for col in df_features.columns if col != 'Activity' and col != 'Person']
    df_features[feature_columns] = scaler.fit_transform(df_features[feature_columns])
    return df_features


def process_all_people():
    """Processes all people and extracts features for walking & jumping."""
    people = ["Alisa", "Brian", "Cissi"]  # Add all available people
    activities = ["Walking", "Jumping"]

    all_data = []

    for person in people:
        for activity in activities:
            df = load_activity_data(person, activity)

            # Split into 5-second windows
            window_size = 100  # Assuming 100 Hz sampling rate
            windows = [df.iloc[i:i + window_size, 1:4].values for i in range(0, len(df), window_size) if
                       len(df.iloc[i:i + window_size]) == window_size]

            # Extract features
            features_list = [extract_features(window) for window in windows]

            # Convert to DataFrame
            df_features = pd.DataFrame(features_list)
            df_features["Activity"] = activity
            df_features["Person"] = person

            all_data.append(df_features)

    # Combine all people’s data
    df_all = pd.concat(all_data, ignore_index=True)

    # Normalize features
    df_all = normalize_features(df_all)

    # Save to CSV
    df_all.to_csv("walking_jumping_features_all_normalized.csv", index=False)

    return df_all


# Run Feature Extraction and Normalization
df_features_normalized = process_all_people()
print("Feature extraction and normalization complete! 🚀")