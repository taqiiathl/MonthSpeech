import os
import joblib
import numpy as np

from mfcc import extract_features

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

DATASET_PATH = "../dataset"

X = []
y = []

print("Loading dataset...")

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".wav"):
            file_path = os.path.join(
                label_path,
                file
            )

            try:
                features = extract_features(
                    file_path
                )

                X.append(features)
                y.append(label)

            except Exception as e:
                print("Error:", file_path)
                print(e)

X = np.array(X)
y = np.array(y)

print(f"Jumlah data : {len(X)}")
print(f"Jumlah fitur : {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Scaling data...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training model...")

model = SVC(
    kernel="rbf",
    probability=True,
    C=10,
    gamma="scale"
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    pred
)

print(f"Akurasi : {acc*100:.2f}%")

print("Classification Report:\n")
print(
    classification_report(
        y_test,
        pred
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        pred
    )
)

joblib.dump(
    model,
    "../models/model_bulan.pkl"
)

joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

print("\nModel tersimpan.")