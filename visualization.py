import matplotlib
matplotlib.use("TkAgg")

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

hdf5_file = "accelerometer_data.h5"
people = ["Alisa", "Brian", "Cissi"]
activities = ["Jumping", "Walking"]

def get_raw_data(person, activity):
    with h5py.File(hdf5_file, "r") as hdf:
        group_path = f"raw_data/{person}/{activity}"
        if group_path not in hdf:
            return None
        group = hdf[group_path]
        all_data = [
            np.array(group[speed][ds])[:, :5]
            for speed in group
            for ds in group[speed]
            if np.array(group[speed][ds]).shape[1] >= 5
        ]
        return pd.DataFrame(np.vstack(all_data), columns=["Time (s)", "Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"]) if all_data else None

def get_preprocessed_data(person, activity):
    with h5py.File(hdf5_file, "r") as hdf:
        group_path = f"preprocessed_data/{person}/{activity}"
        if group_path not in hdf:
            return None
        group = hdf[group_path]
        all_data = [
            np.array(group[speed][ds])[:, :5]
            for speed in group
            for ds in group[speed]
            if np.array(group[speed][ds]).shape[1] >= 5
        ]
        return pd.DataFrame(np.vstack(all_data), columns=["Time (s)", "Accel_X", "Accel_Y", "Accel_Z", "Absolute Accel"]) if all_data else None

#Plotting
def plot_split_axes(activity, data_fn, title_prefix):
    fig, axs = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    fig.suptitle(f"{title_prefix} Accelerometer Data – {activity}", fontsize=18)

    for i, person in enumerate(people):
        df = data_fn(person, activity)
        if df is None:
            for j in range(4):
                axs[i, j].text(0.5, 0.5, f"No Data for {person}", ha='center')
                axs[i, j].set_title(f"{person} - Missing")
            continue

        df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors='coerce')
        df = df[(df["Time (s)"] >= 5.0) & (df["Time (s)"] <= 25.0)]

        axs[i, 0].plot(df["Time (s)"], df["Accel_X"], color="red")
        axs[i, 0].set_title(f"{person} - X")
        axs[i, 1].plot(df["Time (s)"], df["Accel_Y"], color="blue")
        axs[i, 1].set_title(f"{person} - Y")
        axs[i, 2].plot(df["Time (s)"], df["Accel_Z"], color="green")
        axs[i, 2].set_title(f"{person} - Z")
        axs[i, 3].plot(df["Time (s)"], df["Absolute Accel"], color="purple")
        axs[i, 3].set_title(f"{person} - Abs")

        for j in range(4):
            axs[i, j].set_ylabel("Accel (m/s²)")
            axs[i, j].grid(True)

    for ax in axs[-1]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_metadata_summary(activity):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Metadata Summary – {activity}", fontsize=16)

    for i, person in enumerate(people):
        df = get_raw_data(person, activity)
        if df is None:
            axs[i].text(0.5, 0.5, f"No Data for {person}", ha='center')
            axs[i].set_title(person)
            continue

        df = df.apply(pd.to_numeric, errors='coerce')
        stats = {
            'Mean_X': df["Accel_X"].mean(), 'Std_X': df["Accel_X"].std(),
            'Mean_Y': df["Accel_Y"].mean(), 'Std_Y': df["Accel_Y"].std(),
            'Mean_Z': df["Accel_Z"].mean(), 'Std_Z': df["Accel_Z"].std(),
            'Mean_Abs': df["Absolute Accel"].mean(), 'Std_Abs': df["Absolute Accel"].std()
        }

        axs[i].bar(stats.keys(), stats.values(), color=["red", "red", "blue", "blue", "green", "green", "purple", "purple"])
        axs[i].set_title(person)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].set_ylabel("Value")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_individual_axes(person, activity):
    df = get_raw_data(person, activity)
    if df is None:
        print(f"No data for {person} - {activity}")
        return

    df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors='coerce')
    df = df[(df["Time (s)"] >= 5.0) & (df["Time (s)"] <= 25.0)]

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{person} – {activity}", fontsize=16)

    axs[0].plot(df["Time (s)"], df["Accel_X"], color="red"); axs[0].set_ylabel("Accel X")
    axs[1].plot(df["Time (s)"], df["Accel_Y"], color="blue"); axs[1].set_ylabel("Accel Y")
    axs[2].plot(df["Time (s)"], df["Accel_Z"], color="green"); axs[2].set_ylabel("Accel Z")
    axs[3].plot(df["Time (s)"], df["Absolute Accel"], color="purple"); axs[3].set_ylabel("Abs Accel"); axs[3].set_xlabel("Time (s)")

    for ax in axs: ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#Main outputs

#Raw Data – Combined Axis Plots
plot_split_axes("Jumping", get_raw_data, "Raw")
plot_split_axes("Walking", get_raw_data, "Raw")

#Preprocessed Data – Combined Axis Plots
plot_split_axes("Jumping", get_preprocessed_data, "Preprocessed")
plot_split_axes("Walking", get_preprocessed_data, "Preprocessed")

#Raw Metadata
plot_metadata_summary("Jumping")
plot_metadata_summary("Walking")

#Raw Data Per-Person Axis Breakdown
for person in people:
    for activity in activities:
        plot_individual_axes(person, activity)