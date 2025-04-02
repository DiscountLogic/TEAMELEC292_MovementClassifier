import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fft import fft


# -------------------- Feature Extraction --------------------
def extract_features(window):
    features = {}

    for i, axis in enumerate(["Accel_X", "Accel_Y", "Accel_Z"]):
        accel = window[:, i]
        features[f"max_{axis}"] = np.max(accel)
        features[f"min_{axis}"] = np.min(accel)
        features[f"range_{axis}"] = np.ptp(accel)
        features[f"mean_{axis}"] = np.mean(accel)
        features[f"std_{axis}"] = np.std(accel)
        features[f"rms_{axis}"] = np.sqrt(np.mean(accel ** 2))
        features[f"skew_{axis}"] = skew(accel)
        features[f"kurtosis_{axis}"] = kurtosis(accel)
        fft_values = np.abs(fft(accel))
        features[f"fft_peak_{axis}"] = np.argmax(fft_values[1:])  # skip DC
        features[f"fft_mean_{axis}"] = np.mean(fft_values)

    return features


# -------------------- Prediction Pipeline --------------------
from scipy.signal import convolve


# Function for moving average
def moving_average_filter(data, window_size=5):
    return convolve(data, np.ones(window_size) / window_size, mode='same')


def process_and_predict(file_path):
    try:
        df = pd.read_csv(file_path)
        data = df[["Linear Acceleration x (m/s^2)",
                   "Linear Acceleration y (m/s^2)",
                   "Linear Acceleration z (m/s^2)"]].copy()

        data['Absolute Accel'] = np.sqrt(data.iloc[:, 0] ** 2 + data.iloc[:, 1] ** 2 + data.iloc[:, 2] ** 2)

        for col in data.columns:
            data[col] = moving_average_filter(data[col].values, window_size=5)

        if "Time (s)" in df.columns:
            data = data[(df["Time (s)"] >= 5.0) & (df["Time (s)"] <= 25.0)].reset_index(drop=True)
        else:
            data = data.reset_index(drop=True)

        window_size = 500
        windows = [data.iloc[i:i + window_size, :3].values
                   for i in range(0, len(data), window_size)
                   if len(data.iloc[i:i + window_size]) == window_size]

        if not windows:
            messagebox.showerror("Error", "Not enough data for a full 5-second window.")
            return

        features = [extract_features(w) for w in windows]
        df_features = pd.DataFrame(features)

        print("Feature columns before scaling:", df_features.columns.tolist())

        # Load scaler with explicit feature names
        scaler_info = joblib.load("scaler_with_features.pkl")
        scaler = scaler_info['scaler']
        feature_columns = scaler_info['features']

        # Reorder features explicitly
        df_features = df_features[feature_columns]

        print("Scaler means:", scaler.mean_)
        print("Scaler scales:", scaler.scale_)

        df_features_scaled = scaler.transform(df_features)

        model = joblib.load("movement_classifier_model.pkl")
        predictions = model.predict(df_features_scaled)

        label_encoder = LabelEncoder()
        label_encoder.fit(["Jumping", "Walking"])

        predicted_labels = label_encoder.inverse_transform(predictions)

        output_df = pd.DataFrame(predicted_labels, columns=["Predicted Activity"])
        output_file = file_path.replace(".csv", "_predicted.csv")
        output_df.to_csv(output_file, index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(predictions, marker='o', linestyle='-', color='blue')
        plt.title("Predicted Activity per Window")
        plt.xlabel("Window #")
        plt.ylabel("Activity")
        plt.yticks([0, 1], ["Jumping", "Walking"])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("prediction_plot.png")
        print("✅ Plot saved as 'prediction_plot.png'")

        messagebox.showinfo("Success", f"Prediction complete!\nOutput saved as:\n{output_file}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# -------------------- GUI --------------------
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        process_and_predict(file_path)

root = tk.Tk()
root.title("Movement Classifier")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

label = tk.Label(frame, text="Upload accelerometer CSV to classify activity.")
label.pack(pady=10)

upload_button = tk.Button(frame, text="Select File", command=open_file)
upload_button.pack()

root.mainloop()

