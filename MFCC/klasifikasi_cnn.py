import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
print("=== [PROSES 3: TRANING MODEL CNN] ===")

# 1. Memuat Ciri Hasil dari Tahap 2
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print(f"Log: Berhasil memuat {X.shape[0]} data ciri MFCC.")

# 2. Encoding Label (Mengubah teks bulan menjadi angka kategori)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
jumlah_kelas = len(le.classes_)

print("Log: Daftar kelas/bulan yang dideteksi:", le.classes_)

# 3. Reshape agar sesuai format input gambar CNN (Height, Width, Channel)
# Channel = 1 karena MFCC dianggap gambar hitam-putih (Grayscale)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# 4. Split Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
print(f"Log: Data Latih = {X_train.shape[0]}, Data Uji = {X_test.shape[0]}")

# 5. Arsitektur CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(jumlah_kelas, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# OUTPUT LOG: Menampilkan Struktur Lapisan CNN
print("\n--- STRUKTUR ARSITEKTUR CNN ANDA ---")
model.summary()

# 6. Proses Training
print("\n--- MEMULAI PROSES TRAINING MODEL ---")
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# 7. Evaluasi Akhir
print("\n--- HASIL EVALUASI AKHIR ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"🎯 Akurasi Akhir Model CNN Anda: {accuracy * 100:.2f}%")

# Simpan Model Jadian
model.save("model_suara_bulan_cnn.h5")
print("\n✅ TAHAP 3 SELESAI: Model disimpan dengan nama 'model_suara_bulan_cnn.h5'")

print("\n--- MEMBUAT CONFUSION MATRIX ---")

# 1. Lakukan prediksi pada data uji
y_pred_softmax = model.predict(X_test)

# 2. Ubah hasil prediksi (probability/softmax) dan data asli (one-hot) menjadi indeks angka tunggal
y_pred = np.argmax(y_pred_softmax, axis=1)
y_true = np.argmax(y_test, axis=1)

# 3. Hitung Confusion Matrix menggunakan scikit-learn
cm = confusion_matrix(y_true, y_pred)

# 4. Ambil nama-nama kelas asli (nama bulan) dari LabelEncoder untuk label sumbu X dan Y
nama_kelas = le.classes_

# 5. Tampilkan Classification Report (Precision, Recall, F1-Score seperti yang kamu punya di awal)
print("\nClassification Report Lengkap:")
print(classification_report(y_true, y_pred, target_names=nama_kelas))

report = classification_report(
    y_true,
    y_pred,
    target_names=nama_kelas
)

print("\nClassification Report Lengkap:")
print(report)

with open(
    "classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Classification Report\n\n"
    )

    f.write(report)

print(
    "Classification Report disimpan ke classification_report.txt"
)

# 6. Visualisasikan Confusion Matrix dengan Heatmap (Seaborn)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=nama_kelas, yticklabels=nama_kelas)
# 2. Encoding Label (Mengubah teks bulan menjadi angka kategori)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
jumlah_kelas = len(le.classes_)

print("Log: Daftar kelas/bulan yang dideteksi:", le.classes_)

# =====================================================================
# TAMBAHAN: PRINT HASIL X LABEL DAN Y LABEL DALAM FORMAT CSV
# =====================================================================
print("\n=== [LOG DATASET FORMAT CSV] ===")
print("Index,File_Label,Encoded_Label,X_Shape_Asli") # Header CSV

# Menampilkan 10 data pertama sebagai representasi format CSV
for i in range(min(10, len(y))): 
    print(f"{i},{y[i]},{y_encoded[i]},{X[i].shape}")

print(f"... total {len(y)} baris data.")
print("================================\n")
# =====================================================================

plt.title('Confusion Matrix - Pengenalan Suara Nama Bulan')
plt.ylabel('Label Sebenarnya (True Label)')
plt.xlabel('Label Prediksi Model (Predicted Label)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# 7. Tampilkan grafik ke layar
plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Confusion Matrix disimpan ke confusion_matrix.png"
)

plt.show()