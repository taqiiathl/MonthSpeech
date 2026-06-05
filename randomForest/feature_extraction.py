import os
import librosa
import numpy as np
import pandas as pd

# =========================
# DATASET PATH
# =========================
DATASET_PATH = "dataset_clean"

# =========================
# LIST FITUR & LABEL
# =========================
X = []
y = []

# =========================
# LOOP DATASET
# =========================
for label in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, label)

    # pastikan folder
    if os.path.isdir(folder_path):

        # loop file audio
        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                file_path = os.path.join(folder_path, file)

                try:
                    # =========================
                    # LOAD AUDIO
                    # =========================
                    signal, sr = librosa.load(
                        file_path,
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

                    # simpan fitur & label
                    X.append(mfcc_mean)

                    y.append(label)

                    print(f"Processed: {file}")

                except Exception as e:

                    print(f"Error processing {file}")
                    print(e)

# =========================
# NAMA KOLOM
# =========================
columns = [
    f"mfcc_{i+1}"
    for i in range(20)
]

# =========================
# DATAFRAME
# =========================
df = pd.DataFrame(
    X,
    columns=columns
)

# tambah label
df["label"] = y

# =========================
# SIMPAN CSV
# =========================
df.to_csv(
    "features.csv",
    index=False
)

print("\n================================")
print("Feature extraction selesai!")
print("features.csv berhasil dibuat")
print("================================")