from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from gtts import gTTS
import os
import uuid
import time

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

app = Flask(__name__)

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_output")
os.makedirs(AUDIO_DIR, exist_ok=True)


def cleanup_old_files(max_age_seconds=3600):
    """Hapus file audio lama (lebih dari 1 jam)."""
    now = time.time()
    for fname in os.listdir(AUDIO_DIR):
        fpath = os.path.join(AUDIO_DIR, fname)
        if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > max_age_seconds:
            os.remove(fpath)


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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """
    Endpoint utama: terima teks, kecepatan, gender → kembalikan URL audio.
    
    Body JSON:
      - text    : str   — teks yang akan diucapkan
      - speed   : str   — "slow" | "normal" | "fast"
      - gender  : str   — "male" | "female"  (gTTS tidak mendukung gender asli,
                          tapi kita beri opsi untuk ekstensibilitas)
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
        # gTTS hanya punya 2 kecepatan: normal dan slow
        # - slow => slow=True
        # - normal/fast => slow=False
        # Untuk fast kita gunakan normal + playbackRate di browser.
        slow_mode = speed == "slow"

        # Jika gender male dipilih, gunakan pyttsx3 fallback agar suara berbeda.
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


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Sajikan file audio."""
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File tidak ditemukan."}), 404
    ext = os.path.splitext(filename)[1].lower()
    mimetype = "audio/wav" if ext == ".wav" else "audio/mpeg"
    return send_from_directory(AUDIO_DIR, filename, mimetype=mimetype)


@app.route("/download/<filename>")
def download_audio(filename):
    """Unduh file audio."""
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File tidak ditemukan."}), 404
    ext = os.path.splitext(filename)[1] or ".mp3"
    return send_file(filepath, as_attachment=True, download_name=f"tts_output{ext}")


if __name__ == "__main__":
    print("=== TTS App berjalan di http://localhost:5000 ===")
    app.run(debug=True, host="0.0.0.0", port=5000)
