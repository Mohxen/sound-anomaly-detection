import os
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from src.features import extract_logmel
from src.config import SAMPLE_RATE, CHUNK_SIZE


def split_audio(file_path):
    y, sr = sf.read(file_path, dtype="float32")

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != SAMPLE_RATE:
        gcd = np.gcd(sr, SAMPLE_RATE)
        y = resample_poly(y, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)

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


def _is_anomaly_file(path):
    text = " ".join(part.lower() for part in path.parts)
    return any(word in text for word in ("abnormal", "anomaly", "abormal"))


def _features_from_files(files):
    X = []

    for feats in _features_grouped_by_file(files):
        X.extend(feats)

    return np.array(X)


def _features_grouped_by_file(files):
    grouped = []

    for file_path in files:
        chunks = split_audio(str(file_path))
        if not chunks:
            continue

        feats = extract_logmel(np.stack(chunks))
        grouped.append(feats)

    return grouped


def load_normal_train_and_mixed_test(dataset_path, test_ratio=0.2, seed=42):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    wav_files = sorted(dataset_path.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under: {dataset_path}")

    normal_files = [path for path in wav_files if not _is_anomaly_file(path)]
    anomaly_files = [path for path in wav_files if _is_anomaly_file(path)]

    if not normal_files:
        raise ValueError("No normal .wav files found. Put normal files in a folder or filename containing 'normal'.")

    rng = np.random.default_rng(seed)
    normal_files = list(rng.permutation(normal_files))

    if len(normal_files) == 1:
        train_normal_files = normal_files
        test_normal_files = normal_files
    else:
        test_count = max(1, int(round(len(normal_files) * test_ratio)))
        test_count = min(test_count, len(normal_files) - 1)
        test_normal_files = normal_files[:test_count]
        train_normal_files = normal_files[test_count:]

    train_normal_grouped = _features_grouped_by_file(train_normal_files)
    test_normal_grouped = _features_grouped_by_file(test_normal_files)
    test_anomaly_grouped = _features_grouped_by_file(anomaly_files)

    X_train = np.concatenate(train_normal_grouped, axis=0)

    return X_train, train_normal_grouped, test_normal_grouped, test_anomaly_grouped


def load_supervised_file_splits(dataset_path, test_ratio=0.2, seed=42):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    wav_files = sorted(dataset_path.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under: {dataset_path}")

    normal_files = [path for path in wav_files if not _is_anomaly_file(path)]
    anomaly_files = [path for path in wav_files if _is_anomaly_file(path)]

    if not normal_files:
        raise ValueError("No normal .wav files found.")
    if not anomaly_files:
        raise ValueError("No abnormal/anomaly .wav files found.")

    rng = np.random.default_rng(seed)
    normal_files = list(rng.permutation(normal_files))
    anomaly_files = list(rng.permutation(anomaly_files))

    train_normal_files, test_normal_files = _split_files(normal_files, test_ratio)
    train_anomaly_files, test_anomaly_files = _split_files(anomaly_files, test_ratio)

    train_normal_grouped = _features_grouped_by_file(train_normal_files)
    train_anomaly_grouped = _features_grouped_by_file(train_anomaly_files)
    test_normal_grouped = _features_grouped_by_file(test_normal_files)
    test_anomaly_grouped = _features_grouped_by_file(test_anomaly_files)

    X_train = np.concatenate(train_normal_grouped + train_anomaly_grouped, axis=0)
    y_train = np.concatenate(
        [
            np.zeros(sum(len(features) for features in train_normal_grouped), dtype=np.float32),
            np.ones(sum(len(features) for features in train_anomaly_grouped), dtype=np.float32),
        ]
    )

    indices = rng.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    return X_train, y_train, train_normal_grouped, train_anomaly_grouped, test_normal_grouped, test_anomaly_grouped


def _split_files(files, test_ratio):
    if len(files) == 1:
        return files, files

    test_count = max(1, int(round(len(files) * test_ratio)))
    test_count = min(test_count, len(files) - 1)
    return files[test_count:], files[:test_count]
