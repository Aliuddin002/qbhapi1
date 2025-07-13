"""
FastAPI backend for Query‑by‑Humming (QbH)
• Tries local trained_encoder.h5 → falls back to Google Drive download
• Works on Railway, Replit, or any host that may skip Git‑LFS blobs
"""

import os, io, traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn, numpy as np, pandas as pd, librosa, tensorflow as tf

# ─── Config ────────────────────────────────────────────────────────────
SAMPLE_RATE     = 22050
TARGET_SHAPE    = (128, 216)
TOP_N_DEFAULT   = 5
MODEL_PATH      = "trained_encoder.h5"
DRIVE_FILE_ID   = "1xL8fy4lyvjBARw_EXJHt8h0ChZTqaYQX"   # anyone‑with‑link
GDRIVE_URL      = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# ─── Try loading local model; if missing or corrupted, download ────────
def ensure_model():
    from pathlib import Path
    import gdown, shutil, requests

    if Path(MODEL_PATH).is_file() and Path(MODEL_PATH).stat().st_size > 1_000_000:
        print("✅ Local encoder model found.")
        return
    print("⬇️  Encoder model not found or too small → downloading from Google Drive...")
    # Use gdown progress bar
    try:
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    except Exception as e:
        raise RuntimeError(f"❌ gdown failed: {e}")
    if Path(MODEL_PATH).stat().st_size < 1_000_000:
        raise RuntimeError("❌ Downloaded file looks corrupted (size <1 MB).")
    print("🎉 Model downloaded successfully.")

ensure_model()

# ─── Load data ─────────────────────────────────────────────────────────
try:
    encoder        = tf.keras.models.load_model(MODEL_PATH, compile=False)
    features_array = np.load("qbh_features.npy")
    features_index = pd.read_csv("qbh_features_index.csv")
    track_df       = pd.read_csv("track_df_cleaned.csv")
    print("✅ Encoder & feature data loaded.")
except Exception as e:
    print("[FATAL] Could not load model or data.")
    traceback.print_exc()
    raise e

# ─── FastAPI setup ─────────────────────────────────────────────────────
app = FastAPI(title="Query‑by‑Humming API",
              description="Returns top matching tracks from FMA‑small",
              version="1.1.0")

# Helper functions (identical to earlier) … ─────────────────────────────
def audio_to_mel(audio, sr=SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr,
                                         n_mels=TARGET_SHAPE[0], hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    pitches, mags = librosa.piptrack(y=audio, sr=sr, hop_length=512)
    mel_db[:, np.max(mags, 0) > np.percentile(mags, 75)] *= 1.5
    if mel_db.shape[1] < TARGET_SHAPE[1]:
        mel_db = np.pad(mel_db, ((0,0),(0,TARGET_SHAPE[1]-mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :TARGET_SHAPE[1]]
    mel_db -= mel_db.min()
    if mel_db.max() > 0:
        mel_db /= mel_db.max()
    return mel_db

def extract_features(mel):
    mel = mel[np.newaxis, ..., np.newaxis].astype("float32")
    vec = encoder.predict(mel, verbose=0)[0]
    return vec / np.linalg.norm(vec)

def match_tracks(q, top_n=5):
    sims = np.dot(features_array, q)
    sims /= (np.linalg.norm(features_array, 1) * np.linalg.norm(q))
    idx  = sims.argsort()[::-1][:top_n]
    res  = []
    for r,i in enumerate(idx,1):
        tid    = int(features_index.iloc[i]["track_id"])
        row    = track_df[track_df["track_id"]==tid].iloc[0]
        res.append({"rank":r,"track_id":tid,
                    "title":row["title"], "artist":row["artist_name"],
                    "similarity":float(round(sims[i]*100,2))})
    return res

# Routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/qbh")
async def qbh(file:UploadFile=File(...), top_n:int=TOP_N_DEFAULT):
    if file.content_type not in {"audio/wav","audio/mpeg","audio/x-wav",
                                 "audio/wave","audio/flac"}:
        raise HTTPException(415,"Unsupported file type.")
    try:
        y,_ = librosa.load(io.BytesIO(await file.read()),
                           sr=SAMPLE_RATE, mono=True, duration=5)
        if y.size==0: raise ValueError("Empty audio.")
        mel = audio_to_mel(y); vec = extract_features(mel)
        return JSONResponse({"matches": match_tracks(vec, top_n)})
    except Exception as e:
        traceback.print_exc(); raise HTTPException(500,str(e))

if __name__ == "__main__":
    uvicorn.run("qbh_api:app", host="0.0.0.0", port=8000, reload=True)
