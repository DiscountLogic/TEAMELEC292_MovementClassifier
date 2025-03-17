import os
import pandas as pd
import h5py

# Load the CSV file
df = pd.read_csv("Data Collection\Cissi\Jumping\Average\JPSlowJump.csv")

# Display first few rows to confirm structure
print(df.head())

# Define the base folder where CSV files are stored
base_folder = "Data Collection"

# Create an HDF5 file
hdf5_file = h5py.File("accelerometer_data.h5", "w")

# Loop through each person's folder
for person in os.listdir(base_folder):
    person_path = os.path.join(base_folder, person)

    if os.path.isdir(person_path):  # Ensure it's a folder
        person_group = hdf5_file.create_group(f"raw_data/{person}")

        # Loop through activities (Jumping, Walking)
        for activity in os.listdir(person_path):
            activity_path = os.path.join(person_path, activity)

            if os.path.isdir(activity_path):  # Ensure it's a folder
                activity_group = person_group.create_group(activity)

                # Loop through speed categories (Average, Faster)
                for speed in os.listdir(activity_path):
                    speed_path = os.path.join(activity_path, speed)

                    if os.path.isdir(speed_path):  # Ensure it's a folder
                        speed_group = activity_group.create_group(speed)

                        # Loop through all CSV files in the speed folder
                        for csv_file in os.listdir(speed_path):
                            if csv_file.endswith(".csv"):
                                file_path = os.path.join(speed_path, csv_file)

                                # Read CSV file
                                df = pd.read_csv(file_path)

                                # Convert to NumPy array for storage
                                data_array = df.to_numpy()

                                # Store the dataset using the filename as the key
                                dataset_name = csv_file.replace(".csv", "")
                                speed_group.create_dataset(dataset_name, data=data_array)

# Close the HDF5 file
hdf5_file.close()

print("✅ Data successfully stored in HDF5 format!")