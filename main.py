import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

print("Script started...")

# 🔥 DATASET PATH
dataset_path = "dataset/Crema"

emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral"
}

files = os.listdir(dataset_path)[:2500]
print("Total files:", len(files))

X, y = [], []

# 🔥 AUGMENTATION
def augment(signal, sr):
    noise = signal + 0.002 * np.random.randn(len(signal))
    shift = np.roll(signal, int(sr * 0.05))
    return [signal, noise, shift]

max_len = 160

print("Starting feature extraction...")

for i, file in enumerate(files):
    if file.endswith(".wav"):

        if i % 50 == 0:
            print(f"Processing: {i}")

        parts = file.split("_")
        if len(parts) < 3 or parts[2] not in emotion_map:
            continue

        emotion = emotion_map[parts[2]]

        file_path = os.path.join(dataset_path, file)
        signal, sr = librosa.load(file_path, sr=None)

        for sig in augment(signal, sr):

            mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=40)
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            chroma = librosa.feature.chroma_stft(y=sig, sr=sr)

            features = np.vstack((mfcc, delta, delta2, chroma)).T

            if len(features) < max_len:
                features = np.pad(features, ((0, max_len - len(features)), (0, 0)))
            else:
                features = features[:max_len]

            X.append(features)
            y.append(emotion)

print("Feature extraction done!")

# 🔥 MEMORY OPTIMIZATION
X = np.array(X, dtype=np.float32)
y = np.array(y)

# 🔥 NORMALIZATION
X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)

# 🔥 ENCODE LABELS
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 🔥 SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔥 CLASS WEIGHTS
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

print("Starting training...")

# 🚀 MODEL
model = Sequential()

model.add(Conv1D(128, 5, activation='relu', input_shape=(160, X.shape[2])))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))

model.add(Conv1D(256, 5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))

model.add(Dropout(0.4))

model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))

model.add(Dense(6, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    metrics=['accuracy']
)

# 🔥 LR REDUCE
lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4
)

# 🚀 TRAIN (BALANCED)
model.fit(
    X_train, y_train,
    epochs=60,                # 🔥 optimized (fast + high accuracy)
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[lr_reduce],
    class_weight=class_weights,
    shuffle=True
)

# 📊 FINAL EVALUATION
loss, acc = model.evaluate(X_test, y_test)
print("Final Accuracy:", acc)

# 🔍 TRAIN vs TEST
train_acc = model.evaluate(X_train, y_train)[1]
test_acc = model.evaluate(X_test, y_test)[1]

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# 💾 SAVE MODEL
model.save("emotion_model.h5")
print("Model saved successfully!")