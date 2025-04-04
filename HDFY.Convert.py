import os
import pandas as pd
import numpy as np
import h5py

# Define the base folder where CSV files are stored
base_folder = "Data Collection"

# Create an HDF5 file
with h5py.File("accelerometer_data.h5", "w") as hdf5_file:

    # Loop through each person's folder
    for person in os.listdir(base_folder):
        person_path = os.path.join(base_folder, person)
        if not os.path.isdir(person_path):
            continue

        person_group = hdf5_file.create_group(f"raw_data/{person}")

        # Loop through activities (Jumping, Walking)
        for activity in os.listdir(person_path):
            activity_path = os.path.join(person_path, activity)
            if not os.path.isdir(activity_path):
                continue

            activity_group = person_group.create_group(activity)

            # Loop through speed categories (Average, Faster)
            for speed in os.listdir(activity_path):
                speed_path = os.path.join(activity_path, speed)
                if not os.path.isdir(speed_path):
                    continue

                speed_group = activity_group.create_group(speed)

                # Loop through all CSV files in the speed folder
                for csv_file in os.listdir(speed_path):
                    if not csv_file.endswith(".csv"):
                        continue

                    file_path = os.path.join(speed_path, csv_file)

                    # Read CSV file
                    df = pd.read_csv(file_path)

                    # Convert to NumPy array (should be numeric)
                    df = df.apply(pd.to_numeric, errors="coerce")  # Drop this if your data is already numeric
                    df.dropna(axis=0, inplace=True)  # Remove rows with NaNs if they exist

                    data_array = df.to_numpy()

                    # Save to HDF5 using filename (without .csv) as key
                    dataset_name = csv_file.replace(".csv", "")
                    speed_group.create_dataset(dataset_name, data=data_array)

print("Raw data saved to HDF5.")
