import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import pickle
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

print("Starting...")

# ==============================
# PATHS
# ==============================
DATASET_PATH = "dataset/Crema"
OUTPUT_DIR = "cnn_lstm_output"

MAX_LEN = 200
SR = 22050
EPOCHS = 80
BATCH_SIZE = 32
NUM_CLASSES = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# EMOTION MAP
# ==============================
EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral"
}

LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_feature(data, sr, max_len=MAX_LEN):

    min_len = int(sr * 0.5)
    if len(data) < min_len:
        data = np.pad(data, (0, min_len - len(data)))

    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T
    delta = librosa.feature.delta(mfcc.T).T
    delta2 = librosa.feature.delta(mfcc.T, order=2).T
    chroma = librosa.feature.chroma_stft(y=data, sr=sr).T
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=data, sr=sr, n_mels=64)
    ).T
    contrast = librosa.feature.spectral_contrast(y=data, sr=sr).T
    tonnetz = librosa.feature.tonnetz(
        y=librosa.effects.harmonic(data), sr=sr
    ).T
    zcr = librosa.feature.zero_crossing_rate(data).T
    rms = librosa.feature.rms(y=data).T

    min_t = min(len(f) for f in [mfcc, delta, delta2, chroma, mel,
                                contrast, tonnetz, zcr, rms])

    features = np.hstack([
        mfcc[:min_t], delta[:min_t], delta2[:min_t],
        chroma[:min_t], mel[:min_t], contrast[:min_t],
        tonnetz[:min_t], zcr[:min_t], rms[:min_t]
    ])

    if features.shape[0] >= max_len:
        features = features[:max_len]
    else:
        pad = max_len - features.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)))

    return features

# ==============================
# AUGMENTATION
# ==============================
def augment_audio(data, sr):
    augmented = [data]

    augmented.append(data + np.random.randn(len(data)) * 0.005)

    shift = int(np.random.randint(-int(sr * 0.1), int(sr * 0.1)))
    augmented.append(np.roll(data, shift))

    try:
        pitch = librosa.effects.pitch_shift(
            data, sr=sr, n_steps=np.random.uniform(-1, 1)
        )
        augmented.append(pitch)
    except:
        pass

    return augmented

# ==============================
# LOAD DATA
# ==============================
print("Loading dataset...")

all_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".wav")]
print("Total files:", len(all_files))

X, y = [], []

for i, file in enumerate(all_files):

    # 🔥 PROGRESS PRINT
    if i % 100 == 0:
        print(f"Processing {i}/{len(all_files)}")

    parts = file.split("_")
    if len(parts) < 3 or parts[2] not in EMOTION_MAP:
        continue

    emotion = EMOTION_MAP[parts[2]]
    file_path = os.path.join(DATASET_PATH, file)

    try:
        signal, sr = librosa.load(file_path, sr=SR)
    except:
        continue

    for sig in augment_audio(signal, sr):
        feat = extract_feature(sig, sr)

        if feat.shape[0] == MAX_LEN:
            X.append(feat)
            y.append(LABEL_MAP[emotion])

print("Feature extraction done:", len(X))

X = np.array(X, dtype=np.float32)
y = np.array(y)

# ==============================
# ONE HOT
# ==============================
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42
)

# ==============================
# NORMALIZATION
# ==============================
mean = np.mean(X_train, axis=(0, 1), keepdims=True)
std = np.std(X_train, axis=(0, 1), keepdims=True)

X_train = (X_train - mean) / (std + 1e-6)
X_test = (X_test - mean) / (std + 1e-6)

# ==============================
# CLASS WEIGHTS
# ==============================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

class_weights = dict(enumerate(class_weights))

# ==============================
# MODEL
# ==============================
input_layer = Input(shape=(MAX_LEN, X.shape[2]))

x = Conv1D(64, 3, padding='same', activation='relu')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

x = Conv1D(128, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

x = Conv1D(256, 3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.4)(x)

x = Bidirectional(LSTM(64))(x)
x = Dropout(0.4)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.00015),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS
# ==============================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    os.path.join(OUTPUT_DIR, "best_model.keras"),
    monitor="val_accuracy",
    save_best_only=True
)

# ==============================
# TRAIN
# ==============================
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ==============================
# EVALUATE
# ==============================
train_acc = model.evaluate(X_train, y_train)[1]
test_acc = model.evaluate(X_test, y_test)[1]

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Overfitting gap:", train_acc - test_acc)

# ==============================
# SAVE
# ==============================
model.save(os.path.join(OUTPUT_DIR, "emotion_model_crema.h5"))

print("DONE ✅")