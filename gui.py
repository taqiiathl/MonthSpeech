import tkinter as tk
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import joblib
import threading
import os

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model_asr.pkl")

# =========================
# RECORD AUDIO
# =========================
def record_audio():

    # disable tombol saat rekam
    record_button.config(state="disabled")

    status_label.config(text="Merekam suara...")

    fs = 16000
    duration = 2.5

    # =========================
    # REKAM AUDIO
    # =========================
    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    # flatten audio
    recording = recording.flatten()

    # hapus temp lama
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")

    # simpan audio baru
    sf.write("temp.wav", recording, fs)

    status_label.config(text="Memproses audio...")

    # prediksi audio
    predict_audio("temp.wav")

    # aktifkan tombol lagi
    record_button.config(state="normal")

# =========================
# THREADING
# =========================
def start_recording():
    threading.Thread(
        target=record_audio
    ).start()

# =========================
# PREDICT AUDIO
# =========================
def predict_audio(audio_path):

    # =========================
    # LOAD AUDIO
    # =========================
    signal, sr = librosa.load(
        audio_path,
        sr=16000
    )

    # =========================
    # MFCC
    # =========================
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=20
    )

    # =========================
    # RATA-RATA MFCC
    # =========================
    mfcc_mean = np.mean(
        mfcc.T,
        axis=0
    )

    # reshape jadi 2D
    mfcc_mean = mfcc_mean.reshape(1, -1)

    # =========================
    # DATAFRAME
    # =========================
    columns = [
        f"mfcc_{i+1}"
        for i in range(20)
    ]

    mfcc_df = pd.DataFrame(
        mfcc_mean,
        columns=columns
    )

    # =========================
    # PREDIKSI
    # =========================
    prediction = model.predict(mfcc_df)

    probabilities = model.predict_proba(mfcc_df)

    confidence = np.max(probabilities) * 100

    hasil = prediction[0]

    print("\n================================")
    print("Prediction :", hasil)
    print("Confidence :", confidence)

    # =========================
    # OUTPUT GUI
    # =========================
    result_label.config(
        text=f"Bulan: {hasil}"
    )

    status_label.config(
        text=f"Confidence: {confidence:.2f}%"
    )

# =========================
# RESET
# =========================
def reset_result():

    result_label.config(
        text="Bulan: -"
    )

    status_label.config(
        text="Menunggu input..."
    )

# =========================
# GUI
# =========================
root = tk.Tk()

root.title("ASR Nama Bulan")

root.geometry("450x320")

root.resizable(False, False)

# =========================
# TITLE
# =========================
title_label = tk.Label(
    root,
    text="Automatic Speech Recognition",
    font=("Arial", 18, "bold")
)

title_label.pack(pady=20)

# =========================
# BUTTON FRAME
# =========================
button_frame = tk.Frame(root)

button_frame.pack(pady=10)

# tombol rekam
record_button = tk.Button(
    button_frame,
    text="🎤 Mulai Rekam",
    font=("Arial", 13),
    width=15,
    command=start_recording
)

record_button.grid(
    row=0,
    column=0,
    padx=10
)

# tombol reset
reset_button = tk.Button(
    button_frame,
    text="🔄 Reset",
    font=("Arial", 13),
    width=10,
    command=reset_result
)

reset_button.grid(
    row=0,
    column=1,
    padx=10
)

# =========================
# HASIL
# =========================
result_label = tk.Label(
    root,
    text="Bulan: -",
    font=("Arial", 22, "bold"),
    fg="blue"
)

result_label.pack(pady=40)

# =========================
# STATUS
# =========================
status_label = tk.Label(
    root,
    text="Menunggu input...",
    font=("Arial", 11)
)

status_label.pack()

# =========================
# EXIT BUTTON
# =========================
exit_button = tk.Button(
    root,
    text="Keluar",
    font=("Arial", 11),
    width=10,
    command=root.destroy
)

exit_button.pack(pady=20)

# =========================
# RUN GUI
# =========================
root.mainloop()