import os
import librosa
import numpy as np

DATASET_CLEAN = "./dataset_clean"
N_MFCC = 13
MAX_LEN = 50 # Baris kolom waktu

X = []
y = []

print("=== [PROSES 2: EKSTRAKSI CIRI MFCC] ===")

for nama_bulan in os.listdir(DATASET_CLEAN):
    folder_bulan = os.path.join(DATASET_CLEAN, nama_bulan)
    
    if os.path.isdir(folder_bulan):
        print(f"\n🧬 Mengekstrak Ciri untuk Bulan: {nama_bulan}")
        
        for nama_file in os.listdir(folder_bulan):
            if nama_file.endswith('.wav'):
                file_path = os.path.join(folder_bulan, nama_file)
                
                
                audio, sr = librosa.load(file_path, sr=16000)
                trim, _ = librosa.effects.trim(y=audio,top_db=30)
                normals= librosa.util.normalize(trim)
                prem= librosa.effects.preemphasis(normals)


                mfcc = librosa.feature.mfcc(y=prem, sr=sr, n_mfcc=N_MFCC)
                
                bentuk_asal = mfcc.shape
                
                # Padding / Truncating agar ukuran matriks seragam
                if mfcc.shape[1] < MAX_LEN:
                    pad_width = MAX_LEN - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfcc = mfcc[:, :MAX_LEN]
                
                X.append(mfcc)
                y.append(nama_bulan)
                
                
                print(f"   🎵 {nama_file} -> Dimensi Asli: {bentuk_asal} -> Dimensi setelah Padding: {mfcc.shape}")


X = np.array(X)
y = np.array(y)
print(f"============= NILAI X NYA ==============")
print(X)

print(f"============= NILAI Y NYA ==============")
print(y)


np.save("X_features.npy", X)
np.save("y_labels.npy", y)

print("\n✅ TAHAP 2 SELESAI: Ciri suara berhasil diekstrak!")
print(f"   -> Total Gambar Matriks (X): {X.shape}")
print(f"   -> Total Label Suara    (y): {y.shape}")
print("   -> File disimpan: 'X_features.npy' dan 'y_labels.npy'")