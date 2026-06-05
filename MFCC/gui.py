import cv2
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import os

# KONFIGURASI PARAMETER (WAJIB SAMA DENGAN TRAINING)
MAX_LEN = 50
N_MFCC = 13
SAMPLE_RATE = 16000
DURATION = 2  
CLASSES = [
    'Agustus', 'April', 'Desember', 'Februari', 'Januari', 'Juli', 
    'Juni', 'Maret', 'Mei', 'November', 'Oktober', 'September'
]

TEMP_AUDIO_PATH = "temp_live_voice.wav"

# 1. LOAD MODEL .H5
if os.path.exists("model_suara_bulan_cnn.h5"):
    model = tf.keras.models.load_model("model_suara_bulan_cnn.h5")
    status_model = "Model CNN: READY"
else:
    model = None
    status_model = "Model CNN: NOT FOUND!"

# 2. FUNGSI PIPELINE SUARA (BAGIAN YANG DIUBAH/DIREVISI)
def proses_audio_ke_mfcc(file_path):
    # Load audio
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    # --- [TAMBAHAN BARU 1] NORMALISASI VOLUME ---
    # Memastikan suara dari mic yang lemah dinaikkan volumenya
    if np.max(np.abs(audio)) > 0:
        audio = librosa.util.normalize(audio)
        
    # Hapus jeda hening
    audio_clean, _ = librosa.effects.trim(audio, top_db=20)
    
    # --- [TAMBAHAN BARU 2] MENCEGAH AUDIO KOSONG ---
    # Jika hasil trim ternyata menghapus semua suara, gunakan audio aslinya
    if len(audio_clean) == 0:
        audio_clean = audio
        
    # Ekstraksi Ciri MFCC
    mfcc = librosa.feature.mfcc(y=audio_clean, sr=sr, n_mfcc=N_MFCC)
    
    # Padding / Truncating agar ukuran pas (13, 50)
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
        
    return mfcc



canvas = np.zeros((400, 600, 3), dtype="uint8")
window_name = "Aplikasi Klasifikasi Bulan - CNN Realtime"

hasil_prediksi = "Belum Ada Data"
persentase_yakin = "0.00%"
status_rekam = "Tekan SPACE untuk mulai rekam 2 detik"

while True:
    
    canvas.fill(30)

    # Desain Header GUI
    cv2.putText(canvas, "🎙️ AUDIO RECOGNITION (CNN)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(canvas, (30, 70), (570, 70), (100, 100, 100), 1)

    # Info Status Model & Instruksi
    cv2.putText(canvas, f"Status: {status_model}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if model else (0, 0, 255), 1)
    cv2.putText(canvas, status_rekam, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Box Hasil Prediksi
    cv2.rectangle(canvas, (30, 180), (570, 350), (50, 50, 50), -1)
    cv2.rectangle(canvas, (30, 180), (570, 350), (255, 144, 30), 2) # Border Biru Muda
    
    cv2.putText(canvas, "HASIL PREDIKSI CNN:", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, hasil_prediksi.upper(), (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3) # Teks Kuning Besar
    cv2.putText(canvas, f"Keyakinan: {persentase_yakin}", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

    # Footer
    cv2.putText(canvas, "Tekan 'ESC' untuk Keluar Aplikasi", (30, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # Tampilkan Window OpenCV
    cv2.imshow(window_name, canvas)
    
    # Menangkap Inputan Keyboard
    key = cv2.waitKey(1) & 0xFF
    
    # 1. JIKA TEKAN TOMBOL ESC -> KELUAR
    if key == 27: 
        break
        
    # 2. JIKA TEKAN TOMBOL SPACE -> MULAI REKAM REALTIME
    elif key == 32:
        if model is None:
            status_rekam = "ERROR: Model tidak siap!"
            continue
            
        status_rekam = "🎤 SEDANG MEREKAM... BICARALAH SEKARANG!"
        # Menggambar ulang layar instant agar teks perekaman langsung terlihat
        canvas.fill(30)
        cv2.putText(canvas, status_rekam, (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(100)

        try:
            # Proses Perekaman Audio dari Mic Laptop
            rekaman = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait() # Tunggu sampai tepat 2 detik berakhir
            
            # Simpan hasil rekaman mic ke file wav sementara
            write(TEMP_AUDIO_PATH, SAMPLE_RATE, rekaman)
            status_rekam = "Processing ciri MFCC & Memprediksi..."
            
            # JALANKAN PIPELINE KLASIFIKASI CNN
            fitur_mfcc = proses_audio_ke_mfcc(TEMP_AUDIO_PATH)
            input_model = fitur_mfcc.reshape(1, N_MFCC, MAX_LEN, 1)
            
            # # Prediksi oleh file .h5
            # prediksi = model.predict(input_model)
            # indeks_tertinggi = np.argmax(prediksi)
            
            # # Update Variabel untuk Ditampilkan ke GUI
            # hasil_prediksi = CLASSES[indeks_tertinggi]
            
            # Prediksi oleh file .h5
            prediksi = model.predict(input_model)

            print("\n========== HASIL SOFTMAX ==========")

            for i, kelas in enumerate(CLASSES):
                print(
                    f"{kelas:<12} : {prediksi[0][i]:.4f}"
                )

            print("===================================\n")

            indeks_tertinggi = np.argmax(prediksi)

            print(
                "INDEX TERPILIH :",
                indeks_tertinggi
            )

            print(
                "KELAS TERPILIH :",
                CLASSES[indeks_tertinggi]
            )

            hasil_prediksi = CLASSES[indeks_tertinggi]

            persentase_yakin = f"{prediksi[0][indeks_tertinggi] * 100:.2f}%"
            status_rekam = "Selesai! Tekan SPACE lagi untuk mencoba suara baru."
            
        except Exception as e:
            status_rekam = f"Error: {e}"
            
        finally:
            # Hapus file sampah sementara jika ada
            if os.path.exists(TEMP_AUDIO_PATH):
                os.remove(TEMP_AUDIO_PATH)

# Bersihkan semua window setelah keluar loop
cv2.destroyAllWindows()