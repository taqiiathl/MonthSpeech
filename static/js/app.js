document.addEventListener('DOMContentLoaded', () => {
  // ==========================================
  // TAB SWITCHING
  // ==========================================
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Remove active from all
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      
      // Add active to clicked
      btn.classList.add('active');
      const target = document.getElementById(btn.getAttribute('data-target'));
      target.classList.add('active');
    });
  });

  // ==========================================
  // TTS (Text to Speech) LOGIC
  // ==========================================
  const ttsInput = document.getElementById('tts-input');
  const charCount = document.getElementById('charCount');
  const btnSynthesize = document.getElementById('btnSynthesize');
  const ttsOutputArea = document.getElementById('tts-output-area');
  const audioPlayer = document.getElementById('audioPlayer');
  const btnDownload = document.getElementById('btnDownload');
  const ttsError = document.getElementById('ttsError');
  const spinner = btnSynthesize.querySelector('.spinner-small');
  
  let currentAudioUrl = '';

  ttsInput.addEventListener('input', () => {
    charCount.textContent = ttsInput.value.length;
  });

  btnSynthesize.addEventListener('click', async () => {
    const text = ttsInput.value.trim();
    if (!text) {
      showError(ttsError, 'Silakan masukkan teks terlebih dahulu.');
      return;
    }

    const speed = document.querySelector('input[name="speed"]:checked').value;
    const gender = document.querySelector('input[name="gender"]:checked').value;

    // Loading State
    btnSynthesize.disabled = true;
    spinner.hidden = false;
    ttsOutputArea.hidden = true;
    ttsError.hidden = true;

    try {
      const res = await fetch('/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, speed, gender })
      });
      
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.error || 'Terjadi kesalahan');

      currentAudioUrl = data.audio_url;
      audioPlayer.src = currentAudioUrl;
      ttsOutputArea.hidden = false;
      audioPlayer.play();
      
      btnDownload.onclick = () => {
        window.location.href = `/download/${data.filename}`;
      };

    } catch (err) {
      showError(ttsError, err.message);
    } finally {
      btnSynthesize.disabled = false;
      spinner.hidden = true;
    }
  });

  // ==========================================
  // ASR (Speech to Text) LOGIC
  // ==========================================
  const btnRecord = document.getElementById('btnRecord');
  const asrStatus = document.getElementById('asr-status');
  const asrResultArea = document.getElementById('asr-result-area');
  const asrPrediction = document.getElementById('asr-prediction');
  const asrConfidence = document.getElementById('asr-confidence');
  const asrError = document.getElementById('asrError');

  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;

  btnRecord.addEventListener('click', async () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = processAudio;

      mediaRecorder.start();
      isRecording = true;
      
      // UI Updates
      btnRecord.classList.add('recording');
      asrStatus.textContent = 'Merekam... Klik mikrofon lagi untuk berhenti.';
      asrResultArea.hidden = true;
      asrError.hidden = true;

      // Auto stop after 2.5 seconds (based on original ASR script)
      setTimeout(() => {
        if (isRecording) stopRecording();
      }, 2500);

    } catch (err) {
      showError(asrError, 'Gagal mengakses mikrofon. Pastikan izin telah diberikan.');
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    isRecording = false;
    btnRecord.classList.remove('recording');
    asrStatus.textContent = 'Memproses audio...';
  }

  async function processAudio() {
    asrStatus.textContent = 'Memproses format audio...';
    try {
      // 1. Dapatkan file rekaman webm/ogg bawaan browser
      const webmBlob = new Blob(audioChunks);
      const arrayBuffer = await webmBlob.arrayBuffer();

      // 2. Decode menggunakan AudioContext browser untuk mendapatkan raw PCM (otomatis resample ke 16000Hz)
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      // 3. Encode kembali raw PCM menjadi file WAV standar (.wav)
      const wavBlob = audioBufferToWav(audioBuffer);

      // 4. Kirim file WAV tersebut ke server Flask
      const formData = new FormData();
      formData.append('audio', wavBlob, 'recording.wav');

      asrStatus.textContent = 'Memprediksi suara...';
      const res = await fetch('/predict', {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.error || 'Terjadi kesalahan saat memproses audio');

      asrPrediction.textContent = data.prediction;
      asrConfidence.textContent = data.confidence;
      asrResultArea.hidden = false;
      asrStatus.textContent = 'Selesai. Klik mikrofon untuk merekam lagi.';
      
    } catch (err) {
      showError(asrError, err.message);
      asrStatus.textContent = 'Gagal memproses.';
    }
  }

  // Helper
  function showError(element, message) {
    element.textContent = message;
    element.hidden = false;
  }

  // ==========================================
  // WAV ENCODER HELPER (16-bit PCM)
  // ==========================================
  function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    let result;
    if (numChannels === 2) {
      result = interleave(buffer.getChannelData(0), buffer.getChannelData(1));
    } else {
      result = buffer.getChannelData(0);
    }
    
    return encodeWAV(result, format, sampleRate, numChannels, bitDepth);
  }

  function interleave(inputL, inputR) {
    const length = inputL.length + inputR.length;
    const result = new Float32Array(length);
    let index = 0, inputIndex = 0;
    while (index < length) {
      result[index++] = inputL[inputIndex];
      result[index++] = inputR[inputIndex];
      inputIndex++;
    }
    return result;
  }

  function encodeWAV(samples, format, sampleRate, numChannels, bitDepth) {
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
    const view = new DataView(buffer);
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * bytesPerSample, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * bytesPerSample, true);
    floatTo16BitPCM(view, 44, samples);
    
    return new Blob([view], { type: 'audio/wav' });
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function floatTo16BitPCM(view, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, input[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }
});
