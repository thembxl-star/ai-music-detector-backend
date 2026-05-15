from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request, urllib.parse, json, os
import requests as req_lib

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SH_KEY    = os.environ.get("SH_API_KEY", "")
SH_URL    = "https://shlabs.music/api/v1/detect"
DZ        = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def deezer_preview(isrc, artist, track):
    try:
        url = f"https://api.deezer.com/track/isrc:{urllib.parse.quote(isrc)}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=DZ), timeout=8) as r:
            d = json.loads(r.read())
        if d.get("preview"):
            return d["preview"]
    except Exception as e:
        print(f"Deezer ISRC failed: {e}")
    try:
        q = urllib.parse.quote(f'artist:"{artist}" track:"{track}"')
        with urllib.request.urlopen(urllib.request.Request(
                f"https://api.deezer.com/search?q={q}&limit=1", headers=DZ), timeout=8) as r:
            d = json.loads(r.read())
        if d.get("data") and d["data"][0].get("preview"):
            return d["data"][0]["preview"]
    except Exception as e:
        print(f"Deezer search failed: {e}")
    return None

class TrackRequest(BaseModel):
    isrc: str
    artist: str = ""
    track: str = ""

@app.get("/health")
def health():
    return {"status": "ok", "engine": "shlabs", "configured": bool(SH_KEY)}

@app.post("/analyze")
def analyze(req: TrackRequest):
    if not SH_KEY:
        raise HTTPException(500, "SH_API_KEY not set in environment variables")

    print(f"\n--- {req.artist} / {req.track} / {req.isrc} ---")

    # Try ISRC directly first — SHLabs supports it natively
    payload = {"isrc": req.isrc}
    headers = {"X-API-Key": SH_KEY, "Content-Type": "application/json"}

    r = req_lib.post(SH_URL, headers=headers, json=payload, timeout=30)
    print(f"SHLabs (ISRC) {r.status_code}: {r.text[:300]}")

    # If ISRC fails, fall back to Deezer preview URL
    if not r.ok:
        print("ISRC lookup failed, falling back to Deezer preview URL")
        preview = deezer_preview(req.isrc, req.artist, req.track)
        if not preview:
            raise HTTPException(404, "No audio source found — ISRC not on Spotify and no Deezer preview available")
        payload = {"audioUrl": preview}
        r = req_lib.post(SH_URL, headers=headers, json=payload, timeout=30)
        print(f"SHLabs (audioUrl) {r.status_code}: {r.text[:300]}")

    if not r.ok:
        err = r.json() if r.content else {}
        raise HTTPException(r.status_code, err.get("details") or err.get("error") or f"SHLabs error {r.status_code}")

    d = r.json()
    result = d.get("result", {})
    prob = float(result.get("probability_ai_generated", 0))
    prediction = result.get("prediction", "")  # "Human Made", "Pure AI", "Processed AI"
    usage = d.get("usage", {})

    print(f"Result: {prediction} {prob}% — {result.get('most_likely_ai_type','')}")
    print(f"Usage remaining: {usage.get('daily_remaining')} daily / {usage.get('monthly_remaining')} monthly")

    # Deezer preview for playback in UI (best effort)
    preview = deezer_preview(req.isrc, req.artist, req.track)

    return {
        "verdict":          "ai" if prediction in ("Pure AI", "Processed AI") else "human",
        "score":            round(prob / 100, 4),
        "score_pct":        round(prob, 1),
        "prediction":       prediction,
        "confidence":       result.get("confidence_score"),
        "ai_type":          result.get("most_likely_ai_type", ""),
        "spectral":         result.get("spectral_probabilities", {}),
        "temporal":         result.get("temporal_probabilities", {}),
        "preview":          preview,
        "daily_remaining":  usage.get("daily_remaining"),
        "monthly_remaining":usage.get("monthly_remaining"),
    }
