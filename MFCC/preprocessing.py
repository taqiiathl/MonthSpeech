import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os
from audio_recorder_streamlit import audio_recorder


# ==========================================
# KONFIGURASI PARAMETER (WAJIB SAMA DENGAN TRAINING)
# ==========================================
MAX_LEN = 50
N_MFCC = 13

CLASSES = [
    'Agustus', 'April', 'Desember', 'Februari', 'Januari', 'Juli', 
    'Juni', 'Maret', 'Mei', 'November', 'Oktober', 'September'
]

# 1. LOAD MODEL .H5
@st.cache_resource
def load_custom_model():
    if os.path.exists("model_suara_bulan_cnn.h5"):
        return tf.keras.models.load_model("model_suara_bulan_cnn.h5")
    else:
        st.error("File 'model_suara_bulan_cnn.h5' tidak ditemukan! Pastikan file model ada di folder yang sama.")
        return None

model = load_custom_model()

# 2. FUNGSI PIPELINE SUARA
def proses_audio_ke_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    audio_clean, _ = librosa.effects.trim(audio, top_db=20)
    mfcc = librosa.feature.mfcc(y=audio_clean, sr=sr, n_mfcc=N_MFCC)
    
    
    # 1. Trim silence (top_db=30)
    trim, _ = librosa.effects.trim(y=audio, top_db=30)
    
    # Mencegah error jika audio kosong setelah di-trim
    if len(trim) == 0:
        trim = audio
        
    # 2. Normalize
    normals = librosa.util.normalize(trim)
    
    # 3. Preemphasis
    prem = librosa.effects.preemphasis(normals)
    
    # 4. Extract MFCC
    mfcc = librosa.feature.mfcc(y=prem, sr=sr, n_mfcc=N_MFCC)
    
    # 5. Padding/Truncating
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
        
    return mfcc

# ==========================================
# INISIALISASI MEMORI SEMENTARA (SESSION STATE)
# ==========================================
# Ini penting agar suara yang sudah direkam tidak hilang saat tombol klasifikasi diklik
if 'audio_terrekam' not in st.session_state:
    st.session_state.audio_terrekam = None

# ==========================================
# 3. TAMPILKAN GUI UTAMA
# ==========================================
st.title("🎙️ Aplikasi Klasifikasi Bulan (Real-time & Upload)")
st.write("Silakan pilih metode input suara di bawah ini untuk mendeteksi nama bulan.")

# Menu Pilihan Input
metode_input = st.radio("Pilih Metode Input Suara:", ("Rekam Langsung (Real-time)", "Unggah File Audio (.wav)"))

temp_path = "temp_user_voice.wav"

# --- KONDISI 1: JIKA USER MEMILIH REKAM LANGSUNG ---
if metode_input == "Rekam Langsung (Real-time)":
    st.write("Klik ikon mikrofon di bawah untuk mulai merekam, bicara, lalu klik lagi untuk berhenti:")
    
    audio_bytes = audio_recorder(
        text="Klik untuk rekam suara",
        recording_color="#e74c3c",
        neutral_color="#95a5a6",
        icon_size="2x"
    )
    
    if audio_bytes:
        # Simpan hasil rekaman ke dalam memori session state
        st.session_state.audio_terrekam = audio_bytes

# --- KONDISI 2: JIKA USER MEMILIH UPLOAD FILE ---
else:
    uploaded_file = st.file_uploader("Unggah sampel rekaman suara (.wav)", type=["wav"])
    if uploaded_file is not None:
        # Simpan file upload ke dalam memori session state
        st.session_state.audio_terrekam = uploaded_file.read()

# ==========================================
# 4. PROSES KLASIFIKASI (JIKA DATA SUARA ADA)
# ==========================================
if st.session_state.audio_terrekam is not None:
    st.markdown("---")
    st.write("### 🎵 Audio Pilihan Anda:")
    # Putar audio yang tersimpan di memori
    st.audio(st.session_state.audio_terrekam, format='audio/wav')
    
    # Tombol Eksekusi Klasifikasi
    if st.button("Jalankan Klasifikasi 🚀"):
        if model is not None:
            with st.spinner('Model sedang mengekstrak MFCC dan menganalisis suaramu...'):
                
                # Tulis data dari memori ke file fisik sementara agar bisa dibaca Librosa
                with open(temp_path, "wb") as f:
                    f.write(st.session_state.audio_terrekam)
                
                try:
                    # Jalankan Fitur Ekstraksi
                    fitur_mfcc = proses_audio_ke_mfcc(temp_path)
                    input_model = fitur_mfcc.reshape(1, N_MFCC, MAX_LEN, 1)
                    
                    # Prediksi menggunakan model CNN (.h5)
                    prediksi = model.predict(input_model)
                    indeks_tertinggi = np.argmax(prediksi)
                    nama_bulan_terdeteksi = CLASSES[indeks_tertinggi]
                    persentase_yakin = prediksi[0][indeks_tertinggi] * 100
                    
                    # Tampilkan Hasil Utama
                    st.subheader("🏷️ Hasil Analisis CNN:")
                    st.success(f"Model mendeteksi suara: **{nama_bulan_terdeteksi}**")
                    st.info(f"Tingkat Keyakinan Model (Confidence): {persentase_yakin:.2f}%")
                    
                    # Tampilkan Grafik Probabilitas
                    st.write("Grafik Probabilitas Seluruh Kelas Bulan:")
                    st.bar_chart(data=dict(zip(CLASSES, prediksi[0])))
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses audio: {e}")
                
                finally:
                    # Bersihkan file fisik sementara
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
    # Tombol untuk Reset/Hapus rekaman saat ini jika ingin ganti suara baru
    if st.button("🗑️ Hapus/Reset Suara"):
        st.session_state.audio_terrekam = None
        st.rerun()