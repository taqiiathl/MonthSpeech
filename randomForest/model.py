import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# joblib untuk load/save model
# ==========================================
# 1. FUNGSI EKSTRAKSI FITUR
# ==========================================
from utils_audio import extract_audio_features

def extract_features(file_path):
    return extract_audio_features(file_path)

# ==========================================
# 2. PROSES DATA & SIMPAN KE CSV
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "Dataset")
CSV_OUT_PATH = os.path.join(BASE_DIR, "features.csv")

data_rows = []

print("Mulai mengekstrak fitur komprehensif dari audio...")
for month_label in os.listdir(DATASET_PATH):
    month_dir = os.path.join(DATASET_PATH, month_label)
    if os.path.isdir(month_dir):
        for file_name in os.listdir(month_dir):
            if file_name.endswith(('.wav', '.mp3', '.m4a')):
                file_path = os.path.join(month_dir, file_name)
                features_dict = extract_features(file_path)
                if features_dict is not None:
                    features_dict['label_bulan'] = month_label
                    data_rows.append(features_dict)

# Ubah ke Pandas DataFrame
df = pd.DataFrame(data_rows)

# Pindahkan label_bulan ke kolom terakhir
cols = [c for c in df.columns if c != 'label_bulan']
columns = cols # Simpan daftar nama kolom untuk bagian di bawah (feature importance)
df = df[cols + ['label_bulan']]

# SIMPAN KE FILE CSV
df.to_csv(CSV_OUT_PATH, index=False)
print(f"Sukses! Fitur audio telah disimpan di: {CSV_OUT_PATH} dengan {len(columns)} fitur")

# ==========================================
# 3. SPLIT DATA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(df[columns], df['label_bulan'], test_size=0.2, random_state=42, stratify=df['label_bulan'])
print("Hasil Train Data X adalah: ")
print(X_train)
print("Hasil test data X adalah: ")
print(X_test)
# ==========================================
# 4. TRAINING MODEL
# ==========================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

clf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=300,
        max_depth=5,              # Sangat dibatasi agar pohon tidak tumbuh dalam
        min_samples_split=10,     # Harus ada 10 sampel untuk membuat cabang
        min_samples_leaf=5,       # Tiap daun minimal berisi 5 sampel
        max_features=20,          # Memaksa model melihat berbagai fitur, bukan cuma yang terkuat
        random_state=42, 
        n_jobs=-1
    )
)
clf.fit(X_train, y_train)

# Simpan model agar konsisten untuk GUI
MODEL_OUT_PATH = os.path.join(BASE_DIR, "model_randomforest_pipeline.pkl")
joblib.dump(clf, MODEL_OUT_PATH)
print(f"Model disimpan di: {MODEL_OUT_PATH}")

# ==========================================
# 5. MATRIKS EVALUASI & KOEFISIEN (FEATURE IMPORTANCE)
# ==========================================

y_pred = clf.predict(X_test)

print(f"\nAkurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# A. CONFUSION MATRIX (Matriks Tebakan Benar vs Salah)
print("\n[CONFUSION MATRIX]")
labels_order = sorted(df['label_bulan'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
print(cm_df)

# B. FEATURE IMPORTANCE (Matriks Bobot Pengaruh Fitur)
print("\n[MATRIKS KEPENTINGAN FITUR / FEATURE IMPORTANCE (Top 5)]")
# clf adalah Pipeline, ambil step RandomForest yang sesuai
rf = None
for step in getattr(clf, "named_steps", {}).values():
    if hasattr(step, "feature_importances_"):
        rf = step
        break

if rf is None:
    print("Feature importance tidak tersedia karena step RandomForest tidak terdeteksi di pipeline.")
else:
    importances = rf.feature_importances_
    feature_imp_df = pd.DataFrame({'Fitur': columns, 'Tingkat Pengaruh (Koefisien)': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Tingkat Pengaruh (Koefisien)', ascending=False)
    print(feature_imp_df.head(5))

