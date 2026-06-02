import os
import joblib
import numpy as np

from mfcc import extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "model_bulan.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "..",
        "models",
        "scaler.pkl"
    )
)


def predict_audio(audio_path):

    features = extract_features(
        audio_path
    )

    features = features.reshape(
        1,
        -1
    )

    features = scaler.transform(
        features
    )

    prediction = model.predict(
        features
    )[0]

    probabilities = model.predict_proba(
        features
    )[0]

    confidence = np.max(
        probabilities
    )

    print("Prediksi :", prediction)

    for label, prob in zip(
        model.classes_,
        probabilities
    ):
        print(
            f"{label:<10} : {prob:.4f}"
        )

    return prediction, confidence