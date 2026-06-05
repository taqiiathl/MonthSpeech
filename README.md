# Pengenalan Suara Nama Bulan berbasis Deep Learning (CNN & MFCC)

Proyek ini merupakan sistem pengenalan ucapan (speech recognition) otomatis untuk mengklasifikasikan nama-nama bulan dalam bahasa Indonesia (Januari s.d. Desember). Sistem ini menggunakan metode Deep Learning berbasis Convolutional Neural Network (CNN) dengan ekstraksi fitur Mel-Frequency Cepstral Coefficients (MFCC).

Aplikasi diimplementasikan ke dalam dua versi antarmuka: berbasis web menggunakan Streamlit dan aplikasi desktop real-time menggunakan OpenCV. Sistem juga dilengkapi dengan fitur Text-to-Speech (TTS) untuk memberikan umpan balik suara kepada pengguna.

---

## Fitur Utama
* Audio Preprocessing Otomatis: Meliputi resampling (16 kHz), trimming (pemotongan jeda sunyi), normalization, dan pre-emphasis.
* Ekstraksi Fitur Akustik: Menggunakan MFCC dengan 13 koefisien yang diseragamkan ukurannya melalui teknik padding dan truncating.
* Klasifikasi Andal: Menggunakan arsitektur CNN dengan teknik Dropout untuk mencegah overfitting.
* Antarmuka Ganda (Dual-UI): Aplikasi Web berbasis Streamlit dan Aplikasi Desktop Real-time berbasis OpenCV (perekaman langsung menggunakan tombol Space).
* Umpan Balik Suara (TTS): Integrasi Google Text-to-Speech (gTTS) untuk output suara sintetis berbahasa Indonesia.

---

## Alur Sistem & Metodologi

Sistem berjalan melalui tahapan-tahapan terintegrasi berikut:
1. Pengumpulan Dataset: Perekaman audio suara laki-laki berdurasi 1-2 detik berformat .wav.
2. Preprocessing Audio: Standardisasi sinyal agar siap diekstraksi.
3. Ekstraksi Fitur MFCC: Mengubah sinyal audio menjadi representasi matriks numerik ukuran 13 x 50.
4. Pembagian Data: Split dataset menjadi 80% Data Latih (Train) dan 20% Data Uji (Test).
5. Pelatihan CNN: Model mempelajari pola matriks MFCC untuk mengklasifikasikan 12 kelas bulan.
6. Integrasi UI: Model yang telah dilatih (.h5) diintegrasikan ke platform Streamlit dan OpenCV untuk penggunaan langsung.

---

## Detail Spesifikasi Teknis

### 1. Dataset & Parameter Perekaman
* Jumlah Kelas: 12 Kelas (Januari, Februari, ..., Desember).
* Sample Rate Awal: 44.100 Hz -> Resampled: 16.000 Hz.
* Format Audio: WAV.
* Durasi: 1 - 2 detik per sampel.

### 2. Parameter Ekstraksi MFCC
| Parameter | Nilai | Keterangan |
| :--- | :--- | :--- |
| n_mfcc | 13 | Jumlah koefisien MFCC yang dihitung |
| max_len | 50 | Panjang maksimum frame (setelah padding/truncating) |
| Output Ukuran | 13 x 50 | Matriks representasi fitur audio |
| Format Penyimpanan | .npy | Disimpan sebagai X_features.npy and y_labels.npy |

### 3. Arsitektur Model CNN
Model dibangun menggunakan TensorFlow/Keras dengan struktur lapisan sebagai berikut:

| No | Lapisan (Layer) | Parameter | Aktivasi |
| :---: | :--- | :--- | :---: |
| 1 | Conv2D | 32 Filter, Kernel 3 x 3 | ReLU |
| 2 | MaxPooling2D | Pool Size 2 x 2 | - |
| 3 | Dropout | Rate: 0.2 | - |
| 4 | Conv2D | 64 Filter, Kernel 3 x 3 | ReLU |
| 5 | MaxPooling2D | Pool Size 2 x 2 | - |
| 6 | Dropout | Rate: 0.2 | - |
| 7 | Flatten | Mengubah matriks menjadi vektor 1D | - |
| 8 | Dense | 128 Neuron | ReLU |
| 9 | Dropout | Rate: 0.3 | - |
| 10 | Dense (Output) | 12 Neuron (Sesuai jumlah bulan) | Softmax |

**Parameter Pelatihan (Training Parameters):**
* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Epoch: 20
* Batch Size: 16
* Random State Split: 42

---

## Pustaka & Dependensi (Stack Teknologi)

Sistem ini dibangun menggunakan bahasa pemrograman Python dengan library utama:
* tensorflow / keras (Pengembangan model CNN)
* librosa (Prapemrosesan audio & ekstraksi MFCC)
* numpy (Komputasi matriks/array)
* scikit-learn (Evaluasi model & label encoder)
* gTTS (Sintesis suara/Text-to-Speech)
* streamlit (Antarmuka aplikasi web)
* opencv-python (Antarmuka aplikasi desktop real-time)
* sounddevice (Perekaman audio secara real-time)
* matplotlib & seaborn (Visualisasi performa & Confusion Matrix)

---

## Cara Menjalankan Proyek

### 1. Instalasi Dependensi
Pastikan Anda telah menginstal Python (disarankan versi 3.8 - 3.10). Instal semua library yang dibutuhkan melalui terminal/command prompt:
```bash
pip install tensorflow librosa numpy scikit-learn gTTS streamlit opencv-python sounddevice matplotlib seaborn

