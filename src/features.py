import numpy as np
from functools import lru_cache
from scipy.signal import stft
from src.config import SAMPLE_RATE, N_MELS


def _hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def _mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


@lru_cache(maxsize=8)
def _mel_filter(sr, n_fft, n_mels):
    mel_points = np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        if center > left:
            filters[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    return filters


def extract_logmel(y):
    n_fft = 2048
    hop_length = 512

    _, _, spec = stft(
        y,
        fs=SAMPLE_RATE,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
        padded=False,
        axis=-1,
    )
    power = np.abs(spec) ** 2
    mel = np.matmul(_mel_filter(SAMPLE_RATE, n_fft, N_MELS), power)
    logmel = 10 * np.log10(np.maximum(mel, 1e-10))
    return logmel.astype(np.float32)

# def normalize(x):
#     return (x - np.mean(x)) / np.std(x)
