import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Open the HDF5 file
hdf5_file = h5py.File("accelerometer_data.h5", "a")  # Open in append mode

# Create groups for train and test data
if "train_test_split" in hdf5_file:
    del hdf5_file["train_test_split"]  # Remove previous splits if they exist
train_test_group = hdf5_file.create_group("train_test_split")

# Define segment size (5 seconds)
WINDOW_SIZE = 5

# Sampling frequency (assuming ~50Hz, update if different)
SAMPLE_RATE = 50
SAMPLES_PER_WINDOW = WINDOW_SIZE * SAMPLE_RATE

# Access preprocessed data
preprocessed_group = hdf5_file["preprocessed_data"]

for person in preprocessed_group:
    for activity in preprocessed_group[person]:
        for speed in preprocessed_group[person][activity]:
            for position in preprocessed_group[person][activity][speed]:
                dataset = preprocessed_group[person][activity][speed][position]
                data = np.array(dataset)

                # Convert to DataFrame
                df = pd.DataFrame(data, columns=["Time (s)", "Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"])

                # Keep only rows between 5 and 25 seconds
                df = df[(df["Time (s)"] >= 5.0) & (df["Time (s)"] <= 25.0)]

                # Segment the data into 5-second windows
                segmented_data = [df.iloc[i:i + SAMPLES_PER_WINDOW].to_numpy()
                                  for i in range(0, len(df) - SAMPLES_PER_WINDOW, SAMPLES_PER_WINDOW)]

                segmented_data = np.array(segmented_data)  # Convert to numpy array

                # Shuffle and split into 90% training, 10% testing
                train, test = train_test_split(segmented_data, test_size=0.1, random_state=42, shuffle=True)

                # Store in HDF5 file
                train_group = train_test_group.require_group("train")
                test_group = train_test_group.require_group("test")

                train_group.create_dataset(f"{person}/{activity}/{speed}/{position}", data=train)
                test_group.create_dataset(f"{person}/{activity}/{speed}/{position}", data=test)

print("✅ Train-test split complete! Data stored in /train_test_split/")
hdf5_file.close()
