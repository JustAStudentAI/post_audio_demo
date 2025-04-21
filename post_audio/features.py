# post_audio/features.py
"""
Signal-processing helpers: compute a Hann-window FFT & 
return the top-N frequency peaks for each audio file.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq


def fft_peaks(signal : np.ndarray, sr : int, top_n : int = 3):
    """
    Compute the top N frequency peaks in the first 2048 samples.

    signal: the actual audio data (a NumPy array of float values).
    sr: the sample rate (e.g., 16,000 samples per second).
    top_n: the number of strongest frequency peaks you want to return (defaults to 3 if not provided).
    """

    # 1) Slice out the first 2048 samples to keep frame size constant
    frame = signal[:2048]

    # 2) Apply a Hann window
    windowed = frame * np.hanning(len(frame))

    # 3) Compute the one‑sided FFT magnitudes (stores first half of the data)
    mags = np.abs(rfft(windowed))

    # 4) Compute frequencies corresponding to each FFT bin
    freqs = rfftfreq(len(windowed), d=1/sr)

    # 5) Find indices of the top N magnitudes (sorted descending)
    idx = np.argsort(mags)[-top_n:][::-1]

    # 6) Pair each peak frequency with its magnitude, rounded for readability
    peaks = [(round(freqs[i], 1), round(mags[i], 1)) for i in idx]
    
    return peaks