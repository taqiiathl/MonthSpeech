import numpy as np
import librosa

def trim_silence(y: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    if y is None or len(y) == 0:
        return y
    y = np.asarray(y, dtype=np.float32)
    # Gunakan frame length kecil agar pemotongan silence lebih presisi
    trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=512, hop_length=128)
    return trimmed

def extract_audio_features(file_path: str, n_mfcc: int = 20, top_db: float = 20.0) -> dict:
    """
    Ekstrak fitur MFCC dengan Temporal Splitting (dibagi 3 bagian waktu).
    Ini memungkinkan Random Forest 'mendengar' urutan suku kata.
    """
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # 1. NORMALIZE AUDIO VOLUME
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        # 2. TRIM SILENCE 
        # Kembali ke 20 dB agar noise ruangan yang terekam mic GUI benar-benar terpotong
        audio = trim_silence(audio, top_db=top_db)
        
        if len(audio) < 512:
            return None
            
        features = {}
        
        # 3. MFCC TEMPORAL SPLITTING (KUNCI UTAMA!)
        # Daripada merata-rata seluruh suara, kita bagi suara jadi 3 babak (awal, tengah, akhir)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        T = mfccs.shape[1]
        p1, p2 = max(1, T // 3), max(2, 2 * (T // 3))
        
        parts = {
            'awal': mfccs[:, :p1],
            'tengah': mfccs[:, p1:p2],
            'akhir': mfccs[:, p2:]
        }
        
        for part_name, part_data in parts.items():
            if part_data.shape[1] == 0: # Pencegahan error jika audio sangat pendek
                part_data = mfccs
            for i in range(n_mfcc):
                features[f'mfcc_{part_name}_mean_{i}'] = np.mean(part_data[i])
                features[f'mfcc_{part_name}_std_{i}'] = np.std(part_data[i])
                
        # 4. ZCR & RMS (Global)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        rms = librosa.feature.rms(y=audio)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features for {file_path}: {e}")
        return None

