import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import joblib
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load("movement_classifier_model.pkl")
scaler_info = joblib.load("scaler_with_features.pkl")
scaler = scaler_info["scaler"]

# Feature extraction function
def extract_features(window):
    features = {}
    axes = ["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]
    for i, axis in enumerate(axes):
        accel = window[:, i]
        features.update({''
            f"max_{axis}": np.max(accel),
            f"min_{axis}": np.min(accel),
            f"range_{axis}": np.ptp(accel),
            f"mean_{axis}": np.mean(accel),
            f"std_{axis}": np.std(accel),
            f"rms_{axis}": np.sqrt(np.mean(accel ** 2)),
            f"skew_{axis}": skew(accel),
            f"kurtosis_{axis}": kurtosis(accel),
            f"fft_peak_{axis}": np.argmax(np.abs(fft(accel))[1:]),
            f"fft_mean_{axis}": np.mean(np.abs(fft(accel)))
        })
    return features

# CSV classification
def classify_csv(file_path):
    df = pd.read_csv(file_path)
    required_cols = ["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]
    if not all(col in df.columns for col in required_cols):
        messagebox.showerror("Error", "CSV file is missing required columns.")
        return

    window_size = 100  # 5s at 100Hz
    windows = [df.iloc[i:i + window_size][required_cols].values
               for i in range(0, len(df), window_size)
               if len(df.iloc[i:i + window_size]) == window_size]

    features = [extract_features(window) for window in windows]
    df_features = pd.DataFrame(features)
    df_features = scaler.transform(df_features)

    predictions = model.predict(df_features)
    label_map = {0: "Jumping", 1: "Walking"}
    labels = [label_map[p] for p in predictions]

    df_results = pd.DataFrame({"Segment": range(len(labels)), "Predicted Activity": labels})

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Segment"], predictions, marker='o', linestyle='-', label="Predicted Activity")
    plt.title("Predicted Activity Over Segments")
    plt.xlabel("Segment")
    plt.ylabel("Predicted Activity")
    plt.yticks([0, 1], ["Jumping", "Walking"])
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save results
    output_path = file_path.replace(".csv", "_classified.csv")
    df_results.to_csv(output_path, index=False)
    messagebox.showinfo("Success", f"Classification complete! Results saved to:\n{output_path}")

# GUI
def browse_file():
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if path:
        classify_csv(path)

root = tk.Tk()
root.title("Accelerometer Classifier")
root.geometry("400x180")

tk.Label(root, text="Select an accelerometer CSV file to classify:").pack(pady=12)
tk.Button(root, text="Browse", command=browse_file).pack()
root.mainloop()
