# post_audio/features.py
"""
Signal-processing helpers: compute a Hann-window FFT & 
return the top-N frequency peaks for each audio file.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq


def fft_peaks(signal : np.ndarray, sample_rate : int, top_n : int = 3):
    """
    Compute the top N frequency peaks in the first 2048 samples.

    signal: the actual audio data (a NumPy array of float values).
    sample_rate: the sample rate (e.g., 16,000 samples per second).
    top_n: the number of strongest frequency peaks you want to return (defaults to 3 if not provided).
    """
