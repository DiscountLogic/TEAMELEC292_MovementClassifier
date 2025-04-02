#to test the hdf5 file conversion
import h5py

with h5py.File("accelerometer_data.h5", "r") as hdf:
    print("📂 HDF5 File Structure:")
    hdf.visit(print)  # Prints the entire folder structure

#to test the pre processing
# import h5py
#
# with h5py.File("accelerometer_data.h5", "r") as hdf:
#     print("📂 HDF5 Preprocessed Data Structure:")
#     hdf.visit(print)  # Lists all paths in the HDF5 file

#to test the train test split
# import h5py
#
# with h5py.File("accelerometer_data.h5", "r") as hdf:
#     print("📂 HDF5 Train-Test Split Structure:")
#     hdf.visit(print)  # Lists all paths in the HDF5 file

import h5py

# Open the HDF5 file in read mode
# with h5py.File('accelerometer_data.h5', 'r') as hdf:
#     # List all top-level keys (groups) in the HDF5 file
#     print("Keys in HDF5 file:", list(hdf.keys()))
#
#     # Check the structure of the "Segmented data" group
#     if "Segmented data" in hdf:
#         segmented_group = hdf["Segmented data"]
#         print("\n'Next keys in Segmented data':", list(segmented_group.keys()))
#
#         # Check the Train and Test datasets
#         if "Train" in segmented_group:
#             print("\nTrain features and labels:")
#             print("Features shape:", segmented_group["Train/features"].shape)
#             print("Labels shape:", segmented_group["Train/labels"].shape)
#
#         if "Test" in segmented_group:
#             print("\nTest features and labels:")
#             print("Features shape:", segmented_group["Test/features"].shape)
#             print("Labels shape:", segmented_group["Test/labels"].shape)