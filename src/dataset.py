import os
import librosa
import numpy as np
from src.features import extract_logmel
from src.config import SAMPLE_RATE, CHUNK_SIZE


def split_audio(file_path):
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    chunks = []
    for i in range(0, len(y), CHUNK_SIZE):
        chunk = y[i:i+CHUNK_SIZE]
        if len(chunk) == CHUNK_SIZE:
            chunks.append(chunk)
    return chunks


def load_train_data(path):
    X = []

    for file in os.listdir(path):
        if not file.endswith(".wav"):
            continue

        chunks = split_audio(os.path.join(path, file))

        for chunk in chunks:
            feat = extract_logmel(chunk)
            # feat = normalize(feat)

            X.append(feat)

    return np.array(X)


def load_test_data(path):
    normal, anomaly = [], []

    for file in os.listdir(path):
        if not file.endswith(".wav"):
            continue

        chunks = split_audio(os.path.join(path, file))

        for chunk in chunks:
            feat = extract_logmel(chunk)
            if "anomaly" in file:
                anomaly.append(feat)
            else:
                normal.append(feat)

    return np.array(normal), np.array(anomaly)