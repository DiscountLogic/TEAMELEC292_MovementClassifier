import h5py
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

# Load the HDF5 file
with h5py.File('accelerometer_data.h5', 'r') as hdf:
    # Load training data
    X_train = np.array(hdf['Segmented data/Train/features'])
    y_train = np.array(hdf['Segmented data/Train/labels'])

    # Load testing data
    X_test = np.array(hdf['Segmented data/Test/features'])
    y_test = np.array(hdf['Segmented data/Test/labels'])

# Convert labels to integers if they are in string format
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)  # Ensure same encoding for test data

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Logistic Regression model
clf = LogisticRegression(max_iter=1000)

# Learning curve (accuracy over time)
train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate mean and standard deviation for training and testing scores
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training Accuracy', marker='o')
plt.plot(train_sizes, test_mean, label='Testing Accuracy', marker='s')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve: Accuracy Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Train the model on the full training data
clf.fit(X_train, y_train)

# Predictions and probabilities
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)[:, 1]

# Accuracy and recall
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_clf_prob)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.title("ROC Curve")
plt.show()
auc = roc_auc_score(y_test, y_clf_prob)
print(f'AUC: {auc:.4f}')

# Save the trained model
joblib.dump(clf, "movement_classifier_model.pkl")
print("Model saved as 'movement_classifier_model.pkl'")

# Save the scaler with feature names
scaler_info = {"scaler": scaler, "features": [f"Feature_{i}" for i in range(X_train.shape[1])]}  # Generic feature names
joblib.dump(scaler_info, "scaler_with_features.pkl")
print("Scaler saved as 'scaler_with_features.pkl'")