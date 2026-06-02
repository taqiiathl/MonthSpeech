import os
import cv2
import numpy as np
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write
import joblib
from utils_audio import extract_audio_features

# ==========================================
# 1. KONFIGURASI PATH & AUDIO
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset_features.csv") 

SAMPLE_RATE = 44100 
DURATION = 2         # Durasi rekam suara baru (2 detik)
OUTPUT_FILENAME = "temp_live.wav"
WINDOW_NAME = "Audio Classifier GUI"

# ==========================================
# 2. LOAD MODEL PIPELINE
# ==========================================
MODEL_PATH = os.path.join(BASE_DIR, "model_randomforest_pipeline.pkl")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model tidak ditemukan di: {MODEL_PATH}")
    print("Jalankan dulu randomForest/model.py untuk membuat model.")
    exit()

print(f"Memuat model dari: {MODEL_PATH}")
clf = joblib.load(MODEL_PATH)
print("Model siap digunakan!")


# ==========================================
# 3. FUNGSI EKSTRAKSI UNTUK SUARA BARU
# ==========================================
def extract_features_live(file_path):
    """Ekstraksi fitur untuk audio live dan diubah menjadi DataFrame."""
    try:
        # Menggunakan extract_audio_features yang baru dengan banyak fitur
        features_dict = extract_audio_features(file_path)
        
        if features_dict is not None:
            # Dibungkus ke DataFrame agar dibaca lengkap dengan nama fitur oleh Pipeline
            features_df = pd.DataFrame([features_dict])
            return features_df
        return None
    except Exception as e:
        print(f"Error ekstraksi mikrofon: {e}")
        return None


# ==========================================
# 4. FUNGSI DRAW GUI OPENCV
# ==========================================
canvas = np.zeros((400, 600, 3), dtype=np.uint8)

def draw_gui(status_text, prediction_text="", wave_data=None):
    canvas[:] = (35, 35, 35) # Background abu-abu gelap
    
    # Judul & Panduan
    cv2.putText(canvas, "Pendeteksi Nama Bulan (Speech)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "Tekan 'R' untuk Rekam Suara Baru (2 Detik)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, "Tekan 'Q' untuk Keluar", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Status Alat
    color_status = (0, 165, 255) if "Rekam" in status_text else (0, 255, 0)
    cv2.putText(canvas, f"Status : {status_text}", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)
    
    # Hasil Analisis AI
    if prediction_text:
        cv2.putText(canvas, "Hasil Tebakan AI:", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, prediction_text.upper(), (50, 280), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 0), 2)
        
    # Gambar Gelombang Audio (Waveform) jika ada
    if wave_data is not None:
        points = np.linspace(0, len(wave_data) - 1, 500).astype(int)
        for i in range(len(points) - 1):
            x1 = int(50 + (i * 500 / len(points)))
            y1 = int(350 + wave_data[points[i]] * 120) # Skala amplitudo
            x2 = int(50 + ((i + 1) * 500 / len(points)))
            y2 = int(350 + wave_data[points[i+1]] * 120)
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 100), 1)

    cv2.imshow(WINDOW_NAME, canvas)


# ==========================================
# 5. MAIN LOOP GUI
# ==========================================
status = "Ready"
prediksi = ""
audio_waveform = None

while True:
    draw_gui(status, prediksi, audio_waveform)
    
    key = cv2.waitKey(10) & 0xFF
    
    # JIKA USER MENEKAN 'R' UNTUK MEREKAM
    if key == ord('r') or key == ord('R'):
        status = "Merekam... Silakan ucapkan NAMA BULAN!"
        draw_gui(status)
        cv2.waitKey(50) 
        
        # Proses Perekaman Live dari Mikrofon
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait() 
        
        status = "Ekstraksi MFCC & Klasifikasi..."
        draw_gui(status)
        cv2.waitKey(50)
        
        # Simpan audio sementara untuk diekstrak fiturnya
        write(OUTPUT_FILENAME, SAMPLE_RATE, recording)
        audio_waveform = recording.flatten()
        
        # Ekstrak fitur suara live (Seknow mengembalikan DataFrame)
        features_df = extract_features_live(OUTPUT_FILENAME)
        
        if features_df is not None:
            # Prediksi menggunakan model pipeline yang sudah otomatis men-scale data
            prediksi = clf.predict(features_df)[0]
            status = "Selesai menebak!"
        else:
            prediksi = "Error membaca audio mic"
            status = "Gagal"
            
        # Bersihkan file sampah audio
        if os.path.exists(OUTPUT_FILENAME):
            os.remove(OUTPUT_FILENAME)
            
    # JIKA USER MENEKAN 'Q' UNTUK KELUAR
    elif key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()
print("Aplikasi ditutup.")