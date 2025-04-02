import h5py
import numpy as np
import pandas as pd
import scipy.signal as signal

# Open the HDF5 file
hdf5_file = h5py.File("accelerometer_data.h5", "a")  # Open in append mode

# Create a new group for preprocessed data
if "preprocessed_data" in hdf5_file:
    del hdf5_file["preprocessed_data"]  # Remove previous data if exists
preprocessed_group = hdf5_file.create_group("preprocessed_data")

# Define moving average filter function
def moving_average_filter(data, window_size=5):
    return signal.convolve(data, np.ones(window_size) / window_size, mode='same')

# Iterate over raw data and preprocess it
raw_data_group = hdf5_file["raw_data"]

for person in raw_data_group:
    person_group = preprocessed_group.create_group(person)

    for activity in raw_data_group[person]:
        activity_group = person_group.create_group(activity)

        for speed in raw_data_group[person][activity]:
            speed_group = activity_group.create_group(speed)

            for position in raw_data_group[person][activity][speed]:
                dataset = raw_data_group[person][activity][speed][position]
                data = np.array(dataset)

                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(data, columns=["Time (s)",
                                                 "Linear Acceleration x (m/s^2)",
                                                 "Linear Acceleration y (m/s^2)",
                                                 "Linear Acceleration z (m/s^2)",
                                                 "Absolute acceleration (m/s^2)"])

                df = df.ffill()

                # Apply moving average filter using correct names
                df["Linear Acceleration x (m/s^2)"] = moving_average_filter(df["Linear Acceleration x (m/s^2)"])
                df["Linear Acceleration y (m/s^2)"] = moving_average_filter(df["Linear Acceleration y (m/s^2)"])
                df["Linear Acceleration z (m/s^2)"] = moving_average_filter(df["Linear Acceleration z (m/s^2)"])
                df["Absolute acceleration (m/s^2)"] = moving_average_filter(df["Absolute acceleration (m/s^2)"])

                # Store preprocessed data
                speed_group.create_dataset(position, data=df.to_numpy())

print("✅ Preprocessing complete! Data stored in /preprocessed_data/")
hdf5_file.close()