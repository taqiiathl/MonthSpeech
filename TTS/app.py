from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from gtts import gTTS
import edge_tts
import asyncio
import os
import uuid
import time

app = Flask(__name__)

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_output")
os.makedirs(AUDIO_DIR, exist_ok=True)

EDGE_VOICES = {
    "male":   "id-ID-ArdiNeural",
    "female": "id-ID-GadisNeural",
}

RATE_MAP = {
    "slow":   "-20%",
    "normal": "+0%",
    "fast":   "+30%",
}


def cleanup_old_files(max_age_seconds=3600):
    now = time.time()
    for fname in os.listdir(AUDIO_DIR):
        fpath = os.path.join(AUDIO_DIR, fname)
        if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > max_age_seconds:
            os.remove(fpath)


async def synthesize_edge(text, voice, rate, filepath):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(filepath)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(force=True)

    text   = data.get("text", "").strip()
    speed  = data.get("speed", "normal")
    gender = data.get("gender", "female")

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Teks terlalu panjang (maks 5000 karakter)."}), 400

    try:
        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        voice = EDGE_VOICES.get(gender, EDGE_VOICES["female"])
        rate  = RATE_MAP.get(speed, "+0%")

        asyncio.run(synthesize_edge(text, voice, rate, filepath))

        cleanup_old_files()

        return jsonify({
            "audio_url": f"/audio/{filename}",
            "filename":  filename,
            "speed":     speed,
            "gender":    gender,
        })

    except Exception as e:
        return jsonify({"error": f"Gagal menghasilkan suara: {str(e)}"}), 500


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
    print("=== TTS App berjalan di http://localhost:5000 ===")
    app.run(debug=True, host="0.0.0.0", port=5000)
