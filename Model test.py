# simple_test_saved_model.py
import os, csv
import numpy as np
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf

tf.config.optimizer.set_jit(True)
# ---------- settings ----------
TARGET_SIZE = (128, 128)
EPS = 1e-8

# ---------- helpers ----------
def resize_pad(mat, target_rows=128, target_cols=128):
    rows, cols = mat.shape
    # truncate or pad cols
    if cols < target_cols:
        mat = np.pad(mat, ((0,0),(0,target_cols-cols)), mode='constant')
    else:
        mat = mat[:, :target_cols]
    # truncate or pad rows
    if rows < target_rows:
        mat = np.pad(mat, ((0,target_rows-rows),(0,0)), mode='constant')
    else:
        mat = mat[:target_rows, :]
    return mat

def extract_features(file_path):
    # simple inference-only extractor (no augmentation)
    y, sr = librosa.load(file_path, sr=None, mono=True)
    y, _ = librosa.effects.trim(y)
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)), mode='constant')

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)               # (40, t)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)    # (128, t)
    mel_db = librosa.power_to_db(mel, ref=np.max)                   # (128, t)

    # stack (mfcc on top of mel) -> then resize/pad to TARGET_SIZE
    combined = np.vstack([mfcc, mel_db])  # shape (40+128, t)
    combined = resize_pad(combined, target_rows=TARGET_SIZE[0], target_cols=TARGET_SIZE[1])

    # min-max normalize
    mn, mx = combined.min(), combined.max()
    if mx - mn > EPS:
        combined = (combined - mn) / (mx - mn + EPS)
    else:
        combined = np.zeros_like(combined, dtype=np.float32)

    return combined.astype(np.float32)

# ---------- GUI: pick model and audio files ----------
root = tk.Tk(); root.withdraw()

model_path = filedialog.askopenfilename(
    title="Select your saved Keras model (.keras)",
    
)
if not model_path:
    messagebox.showerror("No model", "No model selected. Exiting.")
    raise SystemExit

audio_paths = filedialog.askopenfilenames(
    title="Select audio files to predict (multi-select allowed)",
    filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
)
if not audio_paths:
    messagebox.showerror("No files", "No audio files selected. Exiting.")
    raise SystemExit

print(f"Loaded model: {os.path.basename(model_path)}")
print(f"Files to predict: {len(audio_paths)}")

# ---------- load model ----------
model = tf.keras.models.load_model(model_path)

# ---------- predict ----------
rows = [("filename", "label", "confidence")]
for i, fp in enumerate(audio_paths, 1):
    try:
        feat = extract_features(fp)
        X = feat[np.newaxis, ..., np.newaxis]  # (1,128,128,1)
        prob = float(model.predict(X, verbose=1)[0][0])  # sigmoid -> P(AI)
        label = "AI" if prob >= 0.5 else "HUMAN"
        conf = prob if label == "AI" else 1.0 - prob
        print(f"[{i}/{len(audio_paths)}] {os.path.basename(fp)} -> {label} ({conf*100:.2f}%)")
        rows.append((os.path.basename(fp), label, f"{conf:.4f}"))
    except Exception as e:
        print(f"[{i}/{len(audio_paths)}] {os.path.basename(fp)} -> ERROR: {e}")
        rows.append((os.path.basename(fp), "ERROR", str(e)))

# ---------- save CSV ----------
out_csv = "predictions.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    import csv
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"\nDone — results saved to {out_csv}")
