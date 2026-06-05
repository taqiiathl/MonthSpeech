import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("features.csv")

# fitur
X = df.drop("label", axis=1)

# label
y = df["label"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training :", len(X_train))
print("Testing  :", len(X_test))


# MODEL

model = make_pipeline(
    StandardScaler(),

    RandomForestClassifier(
        n_estimators=500,
        random_state=42
    )
)

# =========================
# TRAINING
# =========================
model.fit(X_train, y_train)

print("\nTraining selesai!")

# =========================
# TESTING
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print(f"Akurasi: {accuracy * 100:.2f}%")
print("================================")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# SIMPAN MODEL
# =========================
joblib.dump(
    model,
    "model_asr.pkl"
)

print("\nModel berhasil disimpan!")