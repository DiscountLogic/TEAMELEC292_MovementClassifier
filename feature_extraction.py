import h5py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

    return df_all

def save_segmented_data_to_hdf5(df_all):
    """Splits data into training and testing sets, and saves it to HDF5."""
    # Split data into features (X) and labels (y)
    X = df_all.drop(columns=["Activity", "Person"]).to_numpy()
    y = df_all["Activity"].to_numpy()

    # Split into training and testing sets (90% train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    with h5py.File(hdf5_file, "a") as hdf:
        # Create 'Segmented data' group if it doesn't exist
        if "Segmented data" not in hdf:
            segmented_data_group = hdf.create_group("Segmented data")
        else:
            segmented_data_group = hdf["Segmented data"]

        # Create 'Train' and 'Test' subgroups
        train_group = segmented_data_group.create_group("Train")
        test_group = segmented_data_group.create_group("Test")

        # Save training and testing data
        train_group.create_dataset("features", data=X_train)
        train_group.create_dataset("labels", data=y_train)

        test_group.create_dataset("features", data=X_test)
        test_group.create_dataset("labels", data=y_test)

    print("Training and testing data saved to HDF5!")

# Run Feature Extraction, Normalization, and Saving to HDF5
df_features_normalized = process_all_people()

# Save the segmented data to HDF5
save_segmented_data_to_hdf5(df_features_normalized)

print("Feature extraction, normalization, segmentation, and saving to HDF5 complete! 🚀")