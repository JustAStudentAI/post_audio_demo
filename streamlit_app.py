# streamlit_app.py
"""Streamlit app for uploading WAV files to display FFT-derived frequency peaks and realtime spectrograms."""

import streamlit as st
import librosa
from post_audio.features import fft_peaks
from post_audio.plots import show_spectrogram

st.title("Post-Audio Spectrogram Viewer")

uploaded = st.file_uploader("Upload a WAV file", type=["wav"])
if uploaded is not None:
    y, sr = librosa.load(uploaded, sr=None)
    st.write(f"Sample rate: **{sr} Hz**")
    st.write(f"Duration: **{len(y)/sr:.2f} s**")

    peaks = fft_peaks(y, sr)
    st.write("Top frequency peaks (Hz, magnitude):", peaks)

    # Generate the Figure and hand it to Streamlit
    fig = show_spectrogram(y, sr)
    st.pyplot(fig)