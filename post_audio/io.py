# post_audio/io.py
"""Functions for loading WAV files, Excel metadata, and notes.txt."""

from pathlib import Path
import pandas as pd
import librosa

def load_wav_file(path : Path):
    # Use sr=None to preserve the original file's sample rate
    y, sample_rate = librosa.load(str(path). sr=None)
    return y, sample_rate

def load_xlsx_metadata(path : Path):
    return pd.read_excel(path)

def load_txt_notes(path : Path):
    """
    Will expect .txt file in format of:
    filename: content
    """
    notes = {}
    with open(path, 'r') as file:
        for line in file:
            if ':' in line:
                file_name, note = line.strip().split(':', 1)
                notes[file_name.strip()] = note.strip()
    return notes