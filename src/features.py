import librosa
import numpy as np
from src.config import SAMPLE_RATE, N_MELS

def extract_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )
    logmel = librosa.power_to_db(mel)
    return logmel

# def normalize(x):
#     return (x - np.mean(x)) / np.std(x)