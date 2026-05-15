from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request, urllib.parse, json, os, time, traceback
import requests as req_lib

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ACR_TOKEN     = os.environ.get("ACR_TOKEN", "")
ACR_CONTAINER = os.environ.get("ACR_CONTAINER", "31424")
ACR_REGION    = os.environ.get("ACR_REGION", "eu-west-1")
DZ            = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

def acr_base():
    return f"https://api-{ACR_REGION}.acrcloud.com/api/fs-containers/{ACR_CONTAINER}"

def acr_auth():
    return {"Authorization": f"Bearer {ACR_TOKEN}", "Accept": "application/json"}

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

def fetch_audio(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=DZ), timeout=15) as r:
        return r.read()

def acr_submit(audio_bytes):
    url = f"{acr_base()}/files"
    print(f"Uploading {len(audio_bytes)} bytes")
    r = req_lib.post(
        url,
        headers=acr_auth(),
        data={"data_type": "audio"},
        files=[("file", ("preview.mp3", audio_bytes, "application/octet-stream"))]
    )
    print(f"Submit {r.status_code}: {r.text[:200]}")
    if not r.ok:
        raise ValueError(f"Submit failed {r.status_code}: {r.text}")
    d = r.json()
    file_id = (d.get("data") or d).get("id")
    if not file_id:
        raise ValueError(f"No file ID: {d}")
    return str(file_id)

def acr_poll(file_id, timeout=120):
    """Poll the list endpoint filtering by file ID until state=1."""
    # The ACRCloud API only has a list endpoint, no single-item GET.
    # We search by file_id and check state.
    url = f"{acr_base()}/files"
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = req_lib.get(url, headers=acr_auth(), params={
            "search": file_id,
            "with_result": 1,
            "per_page": 5
        })
        if not r.ok:
            raise ValueError(f"Poll failed {r.status_code}: {r.text}")
        items = r.json().get("data", [])
        print(f"Poll: {len(items)} items returned")
        # Find our specific file by ID
        match = next((x for x in items if str(x.get("id")) == file_id), None)
        if match is None:
            # Not found yet, still processing
            time.sleep(3)
            continue
        state = match.get("state")
        print(f"state={state} detail={match.get('detail','')}")
        if state == 1:
            print(f"Result: {json.dumps(match)[:500]}")
            return match
        if isinstance(state, int) and state < 0:
            raise ValueError(f"ACRCloud error state={state}: {match.get('detail','')}")
        time.sleep(3)
    raise TimeoutError(f"Timed out waiting for file {file_id}")

def acr_delete(file_id):
    try:
        req_lib.delete(f"{acr_base()}/files/{file_id}", headers=acr_auth(), timeout=10)
        print(f"Deleted {file_id}")
    except Exception as e:
        print(f"Delete failed: {e}")

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
    print(f"\n--- {req.artist} / {req.track} ---")
    preview = deezer_preview(req.isrc, req.artist, req.track)
    if not preview:
        raise HTTPException(404, "No Deezer preview found")
    file_id = None
    try:
        audio = fetch_audio(preview)
        print(f"Downloaded {len(audio)} bytes")
        file_id = acr_submit(audio)
        print(f"File ID: {file_id}")
        data = acr_poll(file_id)
        ai = data.get("results", {}).get("ai_detection", [])
        print(f"AI detection: {ai}")
        if not ai:
            raise HTTPException(422, "No AI detection result — check AI Music Detection is enabled on container 31424")
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
