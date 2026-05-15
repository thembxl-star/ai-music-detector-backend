from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request, urllib.parse, json, os, time
import traceback

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ACR_TOKEN     = os.environ.get("ACR_TOKEN", "")
ACR_CONTAINER = os.environ.get("ACR_CONTAINER", "31424")
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
    except Exception as e:
        print(f"Deezer ISRC lookup failed: {e}")
    try:
        q = urllib.parse.quote(f'artist:"{artist}" track:"{track}"')
        with urllib.request.urlopen(urllib.request.Request(f"https://api.deezer.com/search?q={q}&limit=1", headers=DZ), timeout=8) as r:
            d = json.loads(r.read())
        if d.get("data") and d["data"][0].get("preview"):
            return d["data"][0]["preview"]
    except Exception as e:
        print(f"Deezer search failed: {e}")
    return None

def acr_submit(audio_url, name):
    payload = json.dumps({"data_type": "audio_url", "uri": audio_url, "name": name}).encode()
    url = f"{acr_base()}/files"
    print(f"Submitting to ACR: {url}")
    print(f"Payload: {payload.decode()}")
    req = urllib.request.Request(url, data=payload, headers=acr_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
            print(f"ACR submit response ({r.status}): {raw.decode()}")
            d = json.loads(raw)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"ACR submit HTTP error {e.code}: {body}")
        raise ValueError(f"ACRCloud submit failed {e.code}: {body}")
    file_id = (d.get("data") or d).get("id")
    if not file_id:
        raise ValueError(f"No file ID in response: {d}")
    return str(file_id)

def acr_poll(file_id, timeout=90):
    url = f"{acr_base()}/files/{file_id}?with_result=1"
    print(f"Polling ACR: {url}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=acr_headers()), timeout=15) as r:
                d = json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"ACR poll HTTP error {e.code}: {body}")
            raise ValueError(f"ACRCloud poll failed {e.code}: {body}")
        data = d.get("data", d)
        state = data.get("state")
        print(f"Poll state: {state}")
        if state == 1:
            print(f"Poll result: {json.dumps(data)[:500]}")
            return data
        if isinstance(state, int) and state < 0:
            raise ValueError(f"ACRCloud error state={state}")
        time.sleep(3)
    raise TimeoutError("ACRCloud timed out after 90s")

def acr_delete(file_id):
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{acr_base()}/files/{file_id}",
            headers=acr_headers(), method="DELETE"), timeout=10)
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
        raise HTTPException(500, "ACR_TOKEN not set")
    print(f"\n--- Analyzing: {req.artist} / {req.track} / {req.isrc} ---")
    preview = deezer_preview(req.isrc, req.artist, req.track)
    if not preview:
        raise HTTPException(404, "No Deezer preview found")
    print(f"Preview URL: {preview}")
    file_id = None
    try:
        file_id = acr_submit(preview, f"{req.artist} - {req.track}")
        print(f"File ID: {file_id}")
        result = acr_poll(file_id)
        ai = result.get("results", {}).get("ai_detection", [])
        print(f"AI detection results: {ai}")
        if not ai:
            raise HTTPException(422, "No AI detection result — check container has AI Music Detection enabled")
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
        print(f"ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        if file_id:
            acr_delete(file_id)
