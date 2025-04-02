import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained model
model = joblib.load("movement_classifier_model.pkl")
scaler_info = joblib.load("scaler_with_features.pkl")  # Ensure you save and load the same scaler used during training
scaler = scaler_info["scaler"]
feature_names = scaler_info["features"]

# Feature extraction function
def extract_features(window):
    features = {}
    for i, axis in enumerate( ["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]):
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
        features[f"fft_peak_{axis}"] = np.argmax(fft_values[1:])
        features[f"fft_mean_{axis}"] = np.mean(fft_values)
    return features


# Function to process input CSV and classify
def classify_csv(file_path):
    df = pd.read_csv(file_path)
    required_columns = ["Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)",
                        "Linear Acceleration z (m/s^2)"]

    # Ensure correct columns exist
    if not all(col in df.columns for col in required_columns):
        messagebox.showerror("Error", "CSV file is missing required columns.")
        return

    # Split into 5-second windows (assuming 100 Hz sampling rate)
    window_size = 100
    windows = [df.iloc[i:i + window_size, 1:4].values for i in range(0, len(df), window_size) if
               len(df.iloc[i:i + window_size]) == window_size]

    # Extract features
    features_list = [extract_features(window) for window in windows]
    df_features = pd.DataFrame(features_list)

    # Normalize features
    df_features = scaler.transform(df_features)

    # Predict classes
    predictions = model.predict(df_features)

    # Change output from numeric to string ('Walking' and 'Jumping')
    activity_mapping = {0: "Walking", 1: "Jumping"}
    predicted_activities = [activity_mapping[pred] for pred in predictions]

    # Create results DataFrame
    df_results = pd.DataFrame({"Segment": range(len(predicted_activities)), "Predicted Activity": predicted_activities})

    # Plot the predicted activities
    plt.figure(figsize=(10, 6))
    plt.plot(df_results["Segment"], df_results["Predicted Activity"], label="Predicted Activity", alpha=0.7, marker='o')
    plt.title("Predicted Activity Over Segments")
    plt.xlabel("Segment")
    plt.ylabel("Predicted Activity")
    plt.yticks([0, 1], ["Walking", "Jumping"])  # Map 0 to "Walking" and 1 to "Jumping"
    plt.legend(loc="upper right")
    plt.show()

    # Save results
    output_path = file_path.replace(".csv", "_classified.csv")
    df_results.to_csv(output_path, index=False)
    messagebox.showinfo("Success", f"Classification complete! Results saved to: {output_path}")


# GUI setup
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        classify_csv(file_path)


root = tk.Tk()
root.title("Accelerometer Data Classifier")
root.geometry("400x200")

tk.Label(root, text="Select an accelerometer CSV file:").pack(pady=10)
btn_browse = tk.Button(root, text="Browse", command=browse_file)
btn_browse.pack(pady=10)

root.mainloop()