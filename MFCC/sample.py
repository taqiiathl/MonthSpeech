import os
import time
import sounddevice as sd
from scipy.io.wavfile import write

def record_samples(target_dir, total_samples=50, duration=2, sample_rate=44100):
    # 1. Pastikan folder tujuan sudah ada
    os.makedirs(target_dir, exist_ok=True)
    
    print("=" * 50)
    print(f"Persiapan merekam {total_samples} sampel data voice.")
    print(f"Durasi: {duration} detik per sampel | Jeda: 2 detik")
    print("=" * 50)
    
    
    print("Perekaman akan dimulai dalam 3 detik...")
    time.sleep(3)
    
    for i in range(51, total_samples + 1):
        file_name = f"sample_{i:03d}.wav"
        file_path = os.path.join(target_dir, file_name)
        
     
        print(f"\n[SAMPEL {i}/{total_samples}] 🔴 PEREKAMAN DIMULAI...")
        
    
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()  # Menunggu sampai durasi 2 detik selesai
        
    
        write(file_path, sample_rate, recording)
        print(f"[SAMPEL {i}/{total_samples}] 🟢 SELESAI. Tersimpan sebagai: {file_name}")
        
        
        if i < total_samples:  
            print("TUNGGU YA!!!!!!! Jeda 1 detik... Bersiap untuk sampel berikutnya.")
            time.sleep(1)

    print("\n" + "=" * 50)
    print(f" Selesai! Semua {total_samples} sampel berhasil disimpan di: {target_dir}")
    print("=" * 50)


FOLDER_TUJUAN = 'dataset_clean/April'


record_samples(FOLDER_TUJUAN, total_samples=100, duration=2)