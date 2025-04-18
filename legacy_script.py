# legacy_script.py
'''
Quick script that loads a single .wav file, and 
prints contents of the .xlsx metadata and .txt notes.
'''

import scipy.io.wavfile as wav
import pandas as pd

# Hard‑coded loading of one clip
rate, data = wav.read("data/sound1.wav")
print(f"Loaded clip1.wav @ {rate} Hz → shape {data.shape}")

# Dump metadata from Excel sheet
df = pd.read_excel("data/clip_metadata.xlsx")
print("Metadata head:")
print(df.head())

# Dump notes.txt
with open("data/notes.txt") as f:
    print("Notes:")
    print(f.read())
