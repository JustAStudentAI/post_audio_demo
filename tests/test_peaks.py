# tests/test_peaks.py
"""Unit test for fft peaks."""

import numpy as np
from post_audio.features import fft_peaks

def generate_sine(freq, sr=8000, duration=1.0):
    """Helper: make a pure sine wave at `freq` Hz."""
    t = np.linspace(0, duration, int(sr * duration))
    return np.sin(2 * np.pi * freq * t)

def test_fft_peaks_detects_440hz():
    sr = 8000
    sine = generate_sine(440, sr)
    peaks = fft_peaks(sine, sr, 1)
    detected_freq, _ = peaks[0]
    # Expect within ±5 Hz of 440
    assert abs(detected_freq - 440) < 5

def test_fft_peaks_detects_220hz():
    sr = 8000
    sine = generate_sine(220, sr)
    peaks = fft_peaks(sine, sr, 1)
    detected_freq, _ = peaks[0]
    # Expect within ±5 Hz of 200
    assert abs(detected_freq - 220) < 5

def test_fft_peaks_on_silence():
    sr = 8000
    silence = np.zeros(2048)
    peaks = fft_peaks(silence, sr, 1)
    # magnitude should be zero if no signal
    _, magnitude = peaks[0]
    assert magnitude == 0