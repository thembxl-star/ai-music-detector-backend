from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request, urllib.parse, json, os, time

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ACR_TOKEN     = os.environ.get("ACR_TOKEN", "")
ACR_CONTAINER = os.environ.get("ACR_CONTAINER", "30590")
ACR_REGION    = os.environ.get("ACR_REGION", "eu-west-1")
DZ            = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def acr_headers():
    return {"Authorization": f"Bearer {ACR_TOKEN}", "Accept": "application/json", "Content-Type": "application/json"}

def acr_base():
    return f"https://api-{ACR_REGION}.acrcloud.com/api/fs-containers/{ACR_CONTAINER}"

def deezer_preview(isrc, artist, track):
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
        with urllib.request.urlopen(urllib.request.Request(f"https://api.deezer.com/search?q={q}&limit=1", headers=DZ), timeout=8) as r:
            d = json.loads(r.read())
        if d.get("data") and d["data"][0].get("preview"):
            return d["data"][0]["preview"]
    except Exception:
        pass
    return None

def acr_submit(audio_url, name):
    payload = json.dumps({"data_type": "audio_url", "uri": audio_url, "name": name}).encode()
    req = urllib.request.Request(f"{acr_base()}/files", data=payload, headers=acr_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read())
    file_id = (d.get("data") or d).get("id")
    if not file_id:
        raise ValueError(f"No file ID: {d}")
    return file_id

def acr_poll(file_id, timeout=90):
    url = f"{acr_base()}/files/{file_id}?with_result=1"
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(urllib.request.Request(url, headers=acr_headers()), timeout=15) as r:
            d = json.loads(r.read())
        data = d.get("data", d)
        state = data.get("state")
        if state == 1:
            return data
        if isinstance(state, int) and state < 0:
            raise ValueError(f"ACRCloud error state={state}")
        time.sleep(3)
    raise TimeoutError("ACRCloud timed out")

def acr_delete(file_id):
    try:
        urllib.request.urlopen(urllib.request.Request(f"{acr_base()}/files/{file_id}", headers=acr_headers(), method="DELETE"), timeout=10)
    except Exception:
        pass

class TrackRequest(BaseModel):
    isrc: str
    artist: str = ""
    track: str = ""

@app.get("/health")
def health():
    return {"status": "ok", "acr_configured": bool(ACR_TOKEN), "container": ACR_CONTAINER}

@app.post("/analyze")
def analyze(req: TrackRequest):
    if not ACR_TOKEN:
        raise HTTPException(500, "ACR_TOKEN not set in environment variables")
    preview = deezer_preview(req.isrc, req.artist, req.track)
    if not preview:
        raise HTTPException(404, "No Deezer preview found")
    file_id = None
    try:
        file_id = acr_submit(preview, f"{req.artist} - {req.track}")
        result  = acr_poll(file_id)
        ai      = result.get("results", {}).get("ai_detection", [])
        if not ai:
            raise HTTPException(422, "No AI detection result — enable AI Music Detection on your ACRCloud container")
        det = ai[0]
        prob = float(det.get("ai_probability", 0))
        return {
            "verdict":       "ai" if det.get("prediction") == "ai_generated" else "human",
            "score":         round(prob / 100, 4),
            "score_pct":     round(prob, 1),
            "likely_source": det.get("likely_source", ""),
            "sources":       det.get("source_probabilities", []),
            "preview":       preview,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if file_id:
            acr_delete(file_id)
