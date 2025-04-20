# post_audio/plots.py
"""Visualization helpers: generate and display spectrograms using a 1024-point STFT."""

import numpy as np
import matplotlib.pyplot as plt
import librosa            
import librosa.display

def show_spectrogram(y: np.ndarray, sr: int):
    """
    Compute and display a log-scaled spectrogram for the given audio.
    
    Args:
        y: 1D audio time series.
        sr: Sample rate of `y`.
    """
    HOP_LENGTH = 512

    # 1) Compute STFT: 1024-sample window → freq resolution ≈ sr/1024; 50% overlap
    S = librosa.stft(y, n_fft=1024, hop_length=HOP_LENGTH, window='hann')

    # 2) Convert amplitude to dB scale (logarithmic)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # 3) Create a new figure
    plt.figure(figsize=(6, 3))

    # 4) Draw spectrogram: time on x-axis, log-scaled frequency on y-axis
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        y_axis='log',
        x_axis='time'
    )

    # 5) Add dB colorbar and title
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")

    # 6) Make layout tight and then pop up the window
    plt.tight_layout()
    plt.show()  
    plt.close() 