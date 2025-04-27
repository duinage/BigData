import os
import librosa
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt

# --- Configuration ---
MAD_BASE_DIR = r'military_audio_clustering\data'
ANNOTATION_CSV = r'military_audio_clustering\data\training.csv'
OUTPUT_DIR = r'military_audio_clustering\data\train_melspectrogram_images'
PATH_COL = 'path'
LABEL_COL = 'label'

# Mel Spectrogram Parameters
TARGET_SR = 22050      # Target Sampling Rate to resample audio to
N_FFT = 2048           # FFT window size
HOP_LENGTH = 512       # Hop length for STFT
N_MELS = 128           # Number of Mel bands
FIG_SIZE = (2, 1)      # IMG Resolution
IMG_DPI = 100
# --- End Configuration ---

def create_mel_spectrogram(audio_path, sr, n_fft, hop_length, n_mels):
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    except Exception as e:
        warnings.warn(f"Could not process {audio_path}. Error: {e}")
        return None

def save_spectrogram_image(spec_data, output_path, sr, hop_length, figsize, dpi):
    plt.figure(figsize=figsize)
    librosa.display.specshow(spec_data, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    annotations_df = pd.read_csv(ANNOTATION_CSV)
    for index, row in tqdm(annotations_df.iterrows(), total=annotations_df.shape[0], unit="entry"):
        relative_audio_path = row[PATH_COL]
        label = row[LABEL_COL]

        full_audio_path = os.path.join(MAD_BASE_DIR, relative_audio_path)
        mel_spec_db = create_mel_spectrogram(full_audio_path, TARGET_SR, N_FFT, HOP_LENGTH, N_MELS)

        if mel_spec_db is not None:
            base_filename = os.path.splitext(os.path.basename(relative_audio_path))[0]
            output_filename = f"id{base_filename}_label{label}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            save_spectrogram_image(mel_spec_db, output_path, TARGET_SR, HOP_LENGTH, FIG_SIZE, IMG_DPI)

    print(f"Mel spectrogram images saved to: {os.path.abspath(OUTPUT_DIR)}")