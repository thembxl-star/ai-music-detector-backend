from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request, urllib.parse, json, io, os
import numpy as np
import scipy.signal
import soundfile as sf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_W = None
MODEL_B = None

WEIGHTS_URL = "https://huggingface.co/lofcz/ai-music-detector/resolve/main/weights.npz"

def load_model():
    global MODEL_W, MODEL_B
    cache = "weights_cache.npz"
    if os.path.exists(cache):
        data = np.load(cache)
    else:
        print("Downloading model weights...")
        req = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        with open(cache, "wb") as f:
            f.write(raw)
        data = np.load(io.BytesIO(raw))
    keys = list(data.keys())
    wkey = next(k for k in keys if "weight" in k.lower())
    bkey = next(k for k in keys if "bias" in k.lower())
    MODEL_W = data[wkey].flatten().astype(np.float32)
    MODEL_B = float(data[bkey].flatten()[0])
    print(f"Model ready — {len(MODEL_W)} features, bias={MODEL_B:.4f}")

load_model()

# ── INFERENCE ─────────────────────────────────────────────────────────────────
TARGET_SR  = 16000
FFT_SIZE   = 8192
N_FEATURES = 3585
HULL_AREA  = 10

def extract_fakeprint(audio_bytes: bytes) -> np.ndarray:
    try:
        buf = io.BytesIO(audio_bytes)
        samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_channels(1).set_frame_rate(TARGET_SR)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
        sr = TARGET_SR

    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sr != TARGET_SR:
        samples = scipy.signal.resample_poly(samples, TARGET_SR, sr).astype(np.float32)

    n_frames = len(samples) // FFT_SIZE
    if n_frames < 1:
        raise ValueError("Audio too short")

    win = np.hanning(FFT_SIZE).astype(np.float64)
    avg_mag = np.zeros(FFT_SIZE // 2 + 1, dtype=np.float64)
    for f in range(n_frames):
        frame = samples[f * FFT_SIZE:(f + 1) * FFT_SIZE].astype(np.float64) * win
        avg_mag += np.abs(np.fft.rfft(frame, n=FFT_SIZE))
    avg_mag /= n_frames

    bin_lo = round(1000 * FFT_SIZE / TARGET_SR)
    bin_hi = round(8000 * FFT_SIZE / TARGET_SR)
    slc = avg_mag[bin_lo:bin_hi + 1]

    fp = np.zeros(N_FEATURES, dtype=np.float32)
    for i in range(min(len(slc), N_FEATURES)):
        lo, hi = max(0, i - HULL_AREA), min(len(slc) - 1, i + HULL_AREA)
        fp[i] = max(0.0, float(slc[i]) - float(slc[lo:hi+1].min()))
    return fp

def predict(fp: np.ndarray) -> float:
    logit = MODEL_B + float(np.dot(fp, MODEL_W))
    return float(1 / (1 + np.exp(-logit)))

# ── DEEZER ────────────────────────────────────────────────────────────────────
DZ = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def deezer_preview(isrc: str, artist: str, track: str) -> str | None:
    try:
        url = f"https://api.deezer.com/track/isrc:{urllib.parse.quote(isrc)}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=DZ), timeout=8) as r:
            d = json.loads(r.read())
        if d.get("preview"):
            return d["preview"]
    except Exception:
        pass
    try:
        q = urllib.parse.quote(f'artist:"{artist}" track:"{track}"')
        url = f"https://api.deezer.com/search?q={q}&limit=1"
        with urllib.request.urlopen(urllib.request.Request(url, headers=DZ), timeout=8) as r:
            d = json.loads(r.read())
        if d.get("data") and d["data"][0].get("preview"):
            return d["data"][0]["preview"]
    except Exception:
        pass
    return None

# ── ROUTES ────────────────────────────────────────────────────────────────────
class TrackRequest(BaseModel):
    isrc: str
    artist: str = ""
    track: str = ""

@app.get("/health")
def health():
    return {"status": "ok", "model_features": len(MODEL_W) if MODEL_W is not None else 0}

@app.post("/analyze")
def analyze(req: TrackRequest):
    preview = deezer_preview(req.isrc, req.artist, req.track)
    if not preview:
        raise HTTPException(status_code=404, detail="No Deezer preview found")
    try:
        r = urllib.request.Request(preview, headers=DZ)
        with urllib.request.urlopen(r, timeout=15) as resp:
            audio = resp.read()
        fp    = extract_fakeprint(audio)
        score = predict(fp)
        return {
            "verdict": "ai" if score >= 0.5 else "human",
            "score":   round(score, 4),
            "score_pct": round(score * 100, 1),
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
