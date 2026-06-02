import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(delta, axis=1),
        np.std(delta, axis=1),

        np.mean(delta2, axis=1),
        np.std(delta2, axis=1)
    ])

    return features