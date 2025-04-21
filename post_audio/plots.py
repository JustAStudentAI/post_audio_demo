# post_audio/plots.py
"""Helper function to compute and return spectrogram Figures."""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def show_spectrogram(y: np.ndarray, sr: int) -> plt.Figure:
    """
    Compute and return a log-scaled spectrogram Figure for the given audio.
    (No plt.show(), so it works in Streamlit.)
    """
    HOP_LENGTH = 512

    # 1) Compute STFT
    S = librosa.stft(y, n_fft=2048, hop_length=HOP_LENGTH, window='hann')
    # 2) Convert to dB
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # 3) Create Figure + Axes
    fig, ax = plt.subplots(figsize=(6, 3))

    # 4) Draw spectrogram and capture the mappable
    mesh = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        y_axis='log',
        x_axis='time',
        ax=ax
    )

    # 5) Add colorbar
    fig.colorbar(mesh, ax=ax, format='%+2.0f dB')

    # 6) Title & layout
    ax.set_title("Spectrogram")
    plt.tight_layout()

    return fig