import h5py
import numpy as np
import pandas as pd
import scipy.signal as signal

# Define moving average filter function
def moving_average_filter(data, window_size=5):
    return signal.convolve(data, np.ones(window_size) / window_size, mode='same')

# Open the HDF5 file in append mode
with h5py.File("accelerometer_data.h5", "a") as hdf5_file:

    # Remove old preprocessed data if it exists
    if "preprocessed_data" in hdf5_file:
        del hdf5_file["preprocessed_data"]

    preprocessed_group = hdf5_file.create_group("preprocessed_data")
    raw_data_group = hdf5_file["raw_data"]

    # Iterate through the raw data structure
    for person in raw_data_group:
        person_group = preprocessed_group.create_group(person)

        for activity in raw_data_group[person]:
            activity_group = person_group.create_group(activity)

            for speed in raw_data_group[person][activity]:
                speed_group = activity_group.create_group(speed)

                for position in raw_data_group[person][activity][speed]:
                    dataset = raw_data_group[person][activity][speed][position]
                    data = np.array(dataset)

                    # Convert to DataFrame
                    df = pd.DataFrame(data, columns=["Time (s)", "Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"])
                    df.fillna(method="ffill", inplace=True)  # Handle missing values

                    # Apply moving average filter
                    for col in ["Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"]:
                        df[col] = moving_average_filter(df[col])

                    # Store preprocessed data
                    speed_group.create_dataset(position, data=df.to_numpy())

print("Preprocessing complete.")
