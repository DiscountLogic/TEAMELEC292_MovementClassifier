import h5py
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay

# Load the HDF5 file
with h5py.File('accelerometer_data.h5', 'r') as hdf:
    X = np.array(hdf['Segmented data/Train/features'])
    y = np.array(hdf['Segmented data/Train/labels'])

# Convert labels to integers if they are in string format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and train logistic regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predictions and probabilities
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)[:, 1]

# Print predictions and probabilities
print("Predicted Labels (y_pred):")
print(y_pred)
print("\nPredicted Probabilities (y_clf_prob):")
print(y_clf_prob)

# Accuracy and recall
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_clf_prob)
RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()
auc = roc_auc_score(y_test, y_clf_prob)
print(f'AUC: {auc:.4f}')

# Save the trained model
joblib.dump(clf, "movement_classifier_model.pkl")
print("✅ Model saved as 'movement_classifier_model.pkl'")

# Save the scaler with feature names
scaler_info = {
    "scaler": scaler,
    "features": [f"Feature_{i}" for i in range(X_train.shape[1])]  # Generic feature names
}
joblib.dump(scaler_info, "scaler_with_features.pkl")
print("✅ Scaler saved as 'scaler_with_features.pkl'")