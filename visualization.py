# import matplotlib
# matplotlib.use("TkAgg")  # Fixes the issue with interactive plots
#
# import h5py
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# hdf5_file = "accelerometer_data.h5"
#
#
# def plot_activity_acceleration(person, activity):
#     """Plots acceleration vs. time for a given person and activity (all positions combined)."""
#
#     with h5py.File(hdf5_file, "r") as hdf:
#         preprocessed_data = hdf[f"preprocessed_data/{person}/{activity}"]
#
#         all_data = []
#         for speed in preprocessed_data:
#             for position in preprocessed_data[speed]:
#                 dataset = np.array(preprocessed_data[speed][position])
#                 all_data.append(dataset)
#
#     # Convert to DataFrame
#     data_combined = np.vstack(all_data)  # Stack all position data together
#     df = pd.DataFrame(data_combined, columns=["Time (s)", "Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"])
#
#     # Filter to only keep time from 5 to 25 seconds
#     df = df[(df["Time (s)"] >= 5.0) & (df["Time (s)"] <= 25.0)]
#
#     fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
#
#     # Plot Acceleration X
#     axs[0].plot(df["Time (s)"], df["Accel_X"], label=f"{activity} Accel_X", color="red")
#     axs[0].set_ylabel("Acceleration X (m/s²)")
#     axs[0].set_title(f"{person} - {activity} (All Positions)")
#     axs[0].legend()
#
#     # Plot Acceleration Y
#     axs[1].plot(df["Time (s)"], df["Accel_Y"], label=f"{activity} Accel_Y", color="blue")
#     axs[1].set_ylabel("Acceleration Y (m/s²)")
#     axs[1].legend()
#
#     # Plot Acceleration Z
#     axs[2].plot(df["Time (s)"], df["Accel_Z"], label=f"{activity} Accel_Z", color="green")
#     axs[2].set_ylabel("Acceleration Z (m/s²)")
#     axs[2].legend()
#
#     # Plot Absolute Acceleration (magnitude)
#     axs[3].plot(df["Time (s)"], df["Absolute Accel"], label=f"{activity} Absolute Acceleration", color="purple")
#     axs[3].set_ylabel("Absolute Acceleration (m/s²)")
#     axs[3].set_xlabel("Time (s)")
#     axs[3].legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# # Example usage
# plot_activity_acceleration("Brian", "Jumping")
# plot_activity_acceleration("Brian", "Walking")
# plot_activity_acceleration("Cissi", "Jumping")
# plot_activity_acceleration("Cissi", "Walking")
# plot_activity_acceleration("Alisa", "Jumping")
# plot_activity_acceleration("Alisa", "Walking")

import matplotlib
matplotlib.use("TkAgg")  # Fixes the issue with interactive plots

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hdf5_file = "accelerometer_data.h5"


def plot_activity_acceleration(person, activity):
    """Plots acceleration vs. time for a given person and activity (all positions combined)."""

    with h5py.File(hdf5_file, "r") as hdf:
        preprocessed_data = hdf[f"preprocessed_data/{person}/{activity}"]

        all_data = []
        for speed in preprocessed_data:
            for position in preprocessed_data[speed]:
                dataset = np.array(preprocessed_data[speed][position])
                all_data.append(dataset)

    # Convert to DataFrame
    data_combined = np.vstack(all_data)  # Stack all position data together
    df = pd.DataFrame(data_combined, columns=["Time (s)", "Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"])

    # Filter to only keep time from 5 to 25 seconds
    df = df[(df["Time (s)"] >= 5.0) & (df["Time (s)"] <= 25.0)]

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot Acceleration X
    axs[0].plot(df["Time (s)"], df["Accel_X"], label=f"{activity} Accel_X", color="red")
    axs[0].set_ylabel("Acceleration X (m/s²)")
    axs[0].set_title(f"{person} - {activity} (All Positions)")
    axs[0].legend()

    # Plot Acceleration Y
    axs[1].plot(df["Time (s)"], df["Accel_Y"], label=f"{activity} Accel_Y", color="blue")
    axs[1].set_ylabel("Acceleration Y (m/s²)")
    axs[1].legend()

    # Plot Acceleration Z
    axs[2].plot(df["Time (s)"], df["Accel_Z"], label=f"{activity} Accel_Z", color="green")
    axs[2].set_ylabel("Acceleration Z (m/s²)")
    axs[2].legend()

    # Plot Absolute Acceleration (magnitude)
    axs[3].plot(df["Time (s)"], df["Absolute Accel"], label=f"{activity} Absolute Acceleration", color="purple")
    axs[3].set_ylabel("Absolute Acceleration (m/s²)")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend()

    plt.tight_layout()
    plt.show()


# Example usage
# plot_activity_acceleration("Brian", "Jumping")
# plot_activity_acceleration("Brian", "Walking")
# plot_activity_acceleration("Cissi", "Jumping")
# plot_activity_acceleration("Cissi", "Walking")
# plot_activity_acceleration("Alisa", "Jumping")
# plot_activity_acceleration("Alisa", "Walking")

def plot_meta_data(person, activity):
    """Plots statistical summaries for the accelerometer data."""

    with h5py.File(hdf5_file, "r") as hdf:
        preprocessed_data = hdf[f"preprocessed_data/{person}/{activity}"]

        all_data = []
        for speed in preprocessed_data:
            for position in preprocessed_data[speed]:
                dataset = np.array(preprocessed_data[speed][position])
                all_data.append(dataset)

    # Convert to DataFrame
    data_combined = np.vstack(all_data)
    df = pd.DataFrame(data_combined, columns=["Time (s)", "Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"])

    # Calculate basic statistics for each axis
    stats = {
        'Mean_X': df['Accel_X'].mean(),
        'Std_X': df['Accel_X'].std(),
        'Mean_Y': df['Accel_Y'].mean(),
        'Std_Y': df['Accel_Y'].std(),
        'Mean_Z': df['Accel_Z'].mean(),
        'Std_Z': df['Accel_Z'].std(),
        'Mean_Abs': df['Absolute Accel'].mean(),
        'Std_Abs': df['Absolute Accel'].std(),
    }

    # Create a bar plot to display the statistics
    labels = list(stats.keys())
    values = list(stats.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=['red', 'blue', 'green', 'purple'])
    plt.xticks(rotation=90)
    plt.ylabel("Value")
    plt.title(f"{person} - {activity} Meta Data Summary")
    plt.tight_layout()
    plt.show()

# Example usage for Meta-Data
plot_meta_data("Brian", "Walking")
plot_meta_data("Brian", "Jumping")
plot_meta_data("Cissi", "Walking")
plot_meta_data("Cissi", "Jumping")
plot_meta_data("Alisa", "Walking")
plot_meta_data("Alisa", "Jumping")
