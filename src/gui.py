import tkinter as tk
import sounddevice as sd
import soundfile as sf
import threading
import os
from predict import predict_audio

SAMPLE_RATE = 16000
DURATION = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "..",
    "recordings",
    "temp.wav"
)


def record_and_predict():
    result_label.config(
        text="Merekam..."
    )

    confidence_label.config(
        text=""
    )

    window.update()

    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()
    sf.write(
        OUTPUT_FILE,
        recording,
        SAMPLE_RATE
    )

    result_label.config(
        text="Memproses..."
    )

    window.update()

    try:
        prediction, confidence = predict_audio(
            OUTPUT_FILE
        )

        result_label.config(
            text=f"Hasil : {prediction}"
        )

        confidence_label.config(
            text=f"Confidence : {confidence*100:.2f}%"
        )

    except Exception as e:
        result_label.config(
            text="Terjadi Error"
        )

        confidence_label.config(
            text=str(e)
        )


def start_recording():
    thread = threading.Thread(
        target=record_and_predict
    )
    thread.start()

window = tk.Tk()

window.title(
    "ASR Nama Bulan"
)

window.geometry(
    "600x400"
)

title_label = tk.Label(
    window,
    text="Automatic Speech Recognition\nNama-Nama Bulan",
    font=("Arial", 18, "bold")
)

title_label.pack(
    pady=20
)

instruction_label = tk.Label(
    window,
    text="Klik tombol lalu ucapkan nama bulan",
    font=("Arial", 11)
)

instruction_label.pack()

record_button = tk.Button(
    window,
    text="🎤 REKAM",
    font=("Arial", 14, "bold"),
    width=15,
    command=start_recording
)

record_button.pack(
    pady=20
)

result_label = tk.Label(
    window,
    text="Hasil : -",
    font=("Arial", 16)
)

result_label.pack(
    pady=10
)

confidence_label = tk.Label(
    window,
    text="Confidence : -",
    font=("Arial", 12)
)

confidence_label.pack()

window.mainloop()