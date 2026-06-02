import os
import uuid
import time
import numpy as np
import tensorflow as tf
import librosa
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from gtts import gTTS

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_output")
os.makedirs(AUDIO_DIR, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "MFCC", "model_suara_bulan_cnn.h5")
try:
    asr_model = tf.keras.models.load_model(MODEL_PATH)
    print("ASR CNN Model loaded successfully from develop/MFCC.")
except Exception as e:
    print(f"Error loading ASR CNN Model: {e}")
    asr_model = None

# ==========================================
# CONSTANTS & HELPER FUNCTIONS
# ==========================================
MAX_LEN = 50
N_MFCC = 13
SAMPLE_RATE = 16000
CLASSES = [
    'Agustus', 'April', 'Desember', 'Februari', 'Januari', 'Juli', 
    'Juni', 'Maret', 'Mei', 'November', 'Oktober', 'September'
]

def proses_audio_ke_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
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
def cleanup_old_files(max_age_seconds=3600):
    """Hapus file audio lama (lebih dari 1 jam)."""
    now = time.time()
    for fname in os.listdir(AUDIO_DIR):
        fpath = os.path.join(AUDIO_DIR, fname)
        if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > max_age_seconds:
            try:
                os.remove(fpath)
            except:
                pass

def choose_pyttsx3_voice(gender):
    if pyttsx3 is None:
        return None
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    target_gender = gender.lower()
    for voice in voices:
        voice_gender = getattr(voice, "gender", "").lower()
        if target_gender in voice_gender:
            return voice.id
    return voices[0].id if voices else None

# ==========================================
# ROUTES
# ==========================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """
    Endpoint utama TTS: terima teks, kecepatan, gender → kembalikan URL audio.
    """
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    speed = data.get("speed", "normal")
    gender = data.get("gender", "female")

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Teks terlalu panjang (maks 5000 karakter)."}), 400

    try:
        slow_mode = speed == "slow"

        if gender == "male" and pyttsx3 is not None:
            filename = f"{uuid.uuid4().hex}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            engine = pyttsx3.init()
            voice_id = choose_pyttsx3_voice("male")
            if voice_id:
                engine.setProperty("voice", voice_id)
            base_rate = engine.getProperty("rate") or 200
            rate_map = {"slow": 0.85, "normal": 1.0, "fast": 1.35}
            engine.setProperty("rate", int(base_rate * rate_map.get(speed, 1.0)))
            engine.save_to_file(text, filepath)
            engine.runAndWait()
        else:
            tld = "com"
            if gender == "male":
                tld = "co.id"
            filename = f"{uuid.uuid4().hex}.mp3"
            filepath = os.path.join(AUDIO_DIR, filename)
            tts = gTTS(text=text, lang="id", slow=slow_mode, tld=tld)
            tts.save(filepath)

        cleanup_old_files()

        return jsonify({
            "audio_url": f"/audio/{filename}",
            "filename": filename,
            "speed": speed,
            "gender": gender,
        })
    except Exception as e:
        return jsonify({"error": f"Gagal menghasilkan suara: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint utama ASR: terima file audio wav → kembalikan hasil prediksi teks.
    """
    if asr_model is None:
        return jsonify({"error": "Model ASR tidak dimuat dengan benar."}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "Tidak ada file audio yang diunggah."}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong."}), 400
    
    temp_path = os.path.join(AUDIO_DIR, f"temp_{uuid.uuid4().hex}.wav")
    last_test_path = os.path.join(AUDIO_DIR, "last_test_predict.wav")
    try:
        file.save(temp_path)
        
        # Diagnostics
        audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
        max_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        duration = float(len(audio) / sr) if sr > 0 else 0.0
        print(f"\n[ASR DIAGNOSTIC] Received audio: duration={duration:.2f}s, max_amplitude={max_amp:.4f}, sample_rate={sr}")
        
        # Save copy as last_test_predict.wav
        import shutil
        try:
            shutil.copy2(temp_path, last_test_path)
            print(f"[ASR DIAGNOSTIC] Saved last audio copy to: {last_test_path}")
        except Exception as copy_err:
            print(f"[ASR DIAGNOSTIC] Failed to save copy: {copy_err}")
            
        # Preprocess and Predict with CNN
        fitur_mfcc = proses_audio_ke_mfcc(temp_path)
        input_model = fitur_mfcc.reshape(1, N_MFCC, MAX_LEN, 1)
        
        prediksi = asr_model.predict(input_model)
        
        # Print class probabilities
        print("[ASR DIAGNOSTIC] Prediction probabilities:")
        for i, cl in enumerate(CLASSES):
            print(f"  {cl:<12}: {prediksi[0][i]*100:.2f}%")
            
        indeks_tertinggi = np.argmax(prediksi)
        hasil = CLASSES[indeks_tertinggi]
        confidence = float(prediksi[0][indeks_tertinggi] * 100)
        
        print(f"[ASR DIAGNOSTIC] Result: {hasil} ({confidence:.2f}%)")
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({
            "prediction": str(hasil),
            "confidence": round(float(confidence), 2)
        })
    except Exception as e:
        print(f"[ASR ERROR] Exception: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Gagal memproses audio: {str(e)}"}), 500


@app.route("/audio/<filename>")
def serve_audio(filename):
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File tidak ditemukan."}), 404
    ext = os.path.splitext(filename)[1].lower()
    mimetype = "audio/wav" if ext == ".wav" else "audio/mpeg"
    return send_from_directory(AUDIO_DIR, filename, mimetype=mimetype)


@app.route("/download/<filename>")
def download_audio(filename):
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File tidak ditemukan."}), 404
    ext = os.path.splitext(filename)[1] or ".mp3"
    return send_file(filepath, as_attachment=True, download_name=f"tts_output{ext}")


if __name__ == "__main__":
    print("=== App berjalan di http://localhost:5000 ===")
    app.run(debug=False, host="0.0.0.0", port=5000)
