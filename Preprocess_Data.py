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
                                                 "Accel_X",
                                                 "Accel_Y",
                                                 "Accel_Z",
                                                 "Absolute Accel"])

                # Handle missing values (Forward Fill)
                df.fillna(method="ffill", inplace=True)

                # Apply Moving Average Filter
                df["Accel_X"] = moving_average_filter(df["Accel_X"])
                df["Accel_Y"] = moving_average_filter(df["Accel_Y"])
                df["Accel_Z"] = moving_average_filter(df["Accel_Z"])
                df["Absolute Accel"] = moving_average_filter(df["Absolute Accel"])

                # Store preprocessed data in the HDF5 file
                speed_group.create_dataset(position, data=df.to_numpy())

print("✅ Preprocessing complete! Data stored in /preprocessed_data/")
hdf5_file.close()
