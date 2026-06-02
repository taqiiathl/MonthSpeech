/* ============================================================
   app.js — SuaraKita TTS Frontend Logic
   ============================================================ */

"use strict";

/* ── Elemen DOM ──────────────────────────────────────────────── */
const inputText       = document.getElementById("inputText");
const charCount       = document.getElementById("charCount");
const btnSynthesize   = document.getElementById("btnSynthesize");
const errorMsg        = document.getElementById("errorMsg");
const outputCard      = document.getElementById("outputCard");
const audioPlayer     = document.getElementById("audioPlayer");
const btnDownload     = document.getElementById("btnDownload");
const btnReset        = document.getElementById("btnReset");
const loadingOverlay  = document.getElementById("loadingOverlay");
const outputBadges    = document.getElementById("outputBadges");
const waveformSvg     = document.getElementById("waveformSvg");

/* ── State ───────────────────────────────────────────────────── */
let currentFilename = null;

/* ── Penghitung karakter ──────────────────────────────────────── */
inputText.addEventListener("input", () => {
  const len = inputText.value.length;
  charCount.textContent = len;
  charCount.parentElement.classList.toggle("warn", len > 4800);
});

/* ── Helpers ─────────────────────────────────────────────────── */
function setLoading(active) {
  loadingOverlay.hidden  = !active;
  btnSynthesize.disabled = active;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.hidden = false;
}

function hideError() {
  errorMsg.hidden = true;
  errorMsg.textContent = "";
}

function getSelectedValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}

function labelSpeed(speed) {
  return { slow: "🐢 Lambat", normal: "▶ Normal", fast: "⚡ Cepat" }[speed] || speed;
}

function labelGender(gender) {
  return { female: "👩 Perempuan", male: "👨 Laki-laki" }[gender] || gender;
}

/* ── Waveform dekoratif ──────────────────────────────────────── */
function renderWaveform(active = false) {
  const barCount = 60;
  const w = 400;
  const h = 60;
  const barW = (w / barCount) * 0.65;
  const gap  = (w / barCount) * 0.35;
  const color = active
    ? "var(--color-wave-active)"
    : "var(--color-wave-idle)";

  let bars = "";
  for (let i = 0; i < barCount; i++) {
    // tinggi acak tapi simetris di tengah
    const mid    = barCount / 2;
    const dist   = Math.abs(i - mid) / mid;
    const base   = 10 + (1 - dist) * 35;
    const noise  = (Math.random() - 0.5) * 18;
    const barH   = Math.max(4, base + noise);
    const x      = i * (barW + gap);
    const y      = (h - barH) / 2;

    bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${barH.toFixed(1)}" rx="2" fill="${color}" opacity="${active ? 0.85 : 0.5}"/>`;
  }
  waveformSvg.innerHTML = bars;
}

renderWaveform(false);

/* ── Animasi waveform saat audio play/pause ──────────────────── */
audioPlayer.addEventListener("play",  () => renderWaveform(true));
audioPlayer.addEventListener("pause", () => renderWaveform(false));
audioPlayer.addEventListener("ended", () => renderWaveform(false));

/* ── Synthesize ──────────────────────────────────────────────── */
btnSynthesize.addEventListener("click", async () => {
  hideError();

  const text   = inputText.value.trim();
  const speed  = getSelectedValue("speed")  || "normal";
  const gender = getSelectedValue("gender") || "female";

  if (!text) {
    showError("Harap masukkan teks terlebih dahulu.");
    inputText.focus();
    return;
  }

  setLoading(true);
  outputCard.hidden = true;

  try {
    const res = await fetch("/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, speed, gender }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Terjadi kesalahan pada server.");
    }

    /* Simpan nama file untuk tombol unduh */
    currentFilename = data.filename;

    /* Update audio player */
    audioPlayer.src = data.audio_url + "?t=" + Date.now(); // cache-bust
    audioPlayer.load();
    const playbackRate = {
      slow: 0.85,
      normal: 1.0,
      fast: 1.35,
    }[data.speed] || 1.0;
    audioPlayer.defaultPlaybackRate = playbackRate;
    audioPlayer.playbackRate = playbackRate;
    audioPlayer.play().catch(() => {/* autoplay mungkin diblokir browser */});

    /* Tampilkan badge info */
    outputBadges.innerHTML = `
      <span class="badge">${labelSpeed(data.speed)}</span>
      <span class="badge">${labelGender(data.gender)}</span>
    `;

    /* Tampilkan waveform aktif */
    renderWaveform(true);

    /* Tampilkan output card */
    outputCard.hidden = false;
    outputCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch (err) {
    showError(err.message || "Gagal menghubungi server. Pastikan aplikasi berjalan.");
  } finally {
    setLoading(false);
  }
});

/* ── Unduh ───────────────────────────────────────────────────── */
btnDownload.addEventListener("click", () => {
  if (!currentFilename) return;
  const a = document.createElement("a");
  a.href = `/download/${currentFilename}`;
  a.download = "tts_output.mp3";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

/* ── Reset / Buat Baru ───────────────────────────────────────── */
btnReset.addEventListener("click", () => {
  audioPlayer.pause();
  audioPlayer.src = "";
  outputCard.hidden = true;
  currentFilename = null;
  inputText.value = "";
  charCount.textContent = "0";
  hideError();
  renderWaveform(false);
  inputText.focus();
});

/* ── Shortcut keyboard: Ctrl+Enter / Cmd+Enter ────────────────── */
inputText.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    btnSynthesize.click();
  }
});
