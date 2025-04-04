import h5py
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix,
    roc_curve, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# Load segmented training and testing data
with h5py.File('accelerometer_data.h5', 'r') as hdf:
    X_train = np.array(hdf['Segmented data/Train/features'])
    y_train = np.array(hdf['Segmented data/Train/labels'])
    X_test = np.array(hdf['Segmented data/Test/features'])
    y_test = np.array(hdf['Segmented data/Test/labels'])

# Encode labels if necessary
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression
clf = LogisticRegression(max_iter=1000)

# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train, y_train, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Accuracy', marker='o')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Testing Accuracy', marker='s')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()

# Final model training and evaluation
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.show()

# Save model and scaler
joblib.dump(clf, "movement_classifier_model.pkl")
joblib.dump({"scaler": scaler, "features": [f"Feature_{i}" for i in range(X_train.shape[1])]}, "scaler_with_features.pkl")
