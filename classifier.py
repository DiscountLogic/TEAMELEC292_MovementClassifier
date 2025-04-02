import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib


# Load the HDF5 file
with h5py.File('accelerometer_data.h5', 'r') as hdf:
    # Load the segmented data
    X_train = np.array(hdf['Segmented data/Train/features'])
    y_train = np.array(hdf['Segmented data/Train/labels'])
    X_test = np.array(hdf['Segmented data/Test/features'])
    y_test = np.array(hdf['Segmented data/Test/labels'])

# Convert labels to integers if they are in string format
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Save the scaler to use in the desktop app
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as 'scaler.pkl'")

# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Use learning_curve to get training and test scores for various training set sizes
train_sizes, train_scores, test_scores = learning_curve(logreg, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

# Calculate the mean and standard deviation of the training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Print accuracy values for each training set size
print("Training and Test Accuracy for Different Training Set Sizes:")
for i, size in enumerate(train_sizes):
    print(f"Training Size: {size:.2f}")
    print(f"  Training Accuracy: {train_mean[i]:.4f} ± {train_std[i]:.4f}")
    print(f"  Test Accuracy: {test_mean[i]:.4f} ± {test_std[i]:.4f}")
    print()

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', color='blue')
plt.plot(train_sizes, test_mean, label='Test Accuracy', color='green')

# Plot the standard deviation as shaded areas
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='green')

plt.title('Learning Curve (Training and Test Accuracy)')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

#CHANGED PLT.SHOW WITH BELOEW:
plt.savefig("learning_curve.png")
print("✅ Learning curve saved as 'learning_curve.png'")


# save the train logistic regression model

import joblib

joblib.dump(logreg, "movement_classifier_model.pkl")
print("✅ Model saved as 'movement_classifier_model.pkl'")

print("LabelEncoder classes:", label_encoder.classes_)

