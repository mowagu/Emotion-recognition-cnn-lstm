import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

print("Loading model...")

model = load_model("emotion_model.h5")

labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral']
max_len = 160

def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)

    features = np.vstack((mfcc, delta, delta2, chroma)).T

    if len(features) < max_len:
        features = np.pad(features, ((0, max_len - len(features)), (0, 0)))
    else:
        features = features[:max_len]

    features = (features - np.mean(features)) / (np.std(features) + 1e-8)

    return np.expand_dims(features, axis=0)


def predict_emotion(file_path):
    features = extract_features(file_path)
    prediction = model.predict(features)
    return labels[np.argmax(prediction)]


file = input("Enter audio file name: ")


print("Predicting...")

result = predict_emotion(file)
print("Predicted Emotion:", result)