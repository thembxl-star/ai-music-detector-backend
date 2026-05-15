from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request, urllib.parse, json, os, time, traceback

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
        with urllib.request.urlopen(urllib.request.Request(f"https://api.deezer.com/search?q={q}&limit=1", headers=DZ), timeout=8) as r:
            d = json.loads(r.read())
        if d.get("data") and d["data"][0].get("preview"):
            return d["data"][0]["preview"]
    except Exception as e:
        print(f"Deezer search failed: {e}")
    return None

def fetch_audio(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=DZ), timeout=15) as r:
        return r.read()

def build_multipart(fields, files, boundary):
    """Build multipart/form-data body.
    fields: dict of {name: value}
    files: dict of {name: (filename, content_type, bytes)}
    """
    body = b""
    for name, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    for name, (filename, content_type, data) in files.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body

def acr_submit(audio_bytes, name):
    boundary = "----ACRFormBoundary7MA4YWxkTrZu0gW"
    body = build_multipart(
        fields={"data_type": "audio", "name": name},
        files={"audio_file": ("preview.mp3", "audio/mpeg", audio_bytes)},
        boundary=boundary
    )
    url = f"{acr_base()}/files"
    print(f"Uploading {len(audio_bytes)} bytes to {url}")
    headers = {**acr_auth(), "Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            print(f"ACR submit OK: {raw.decode()[:300]}")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"ACR submit error {e.code}: {err}")
        raise ValueError(f"ACRCloud {e.code}: {err}")

def acr_poll(file_id, timeout=90):
    url = f"{acr_base()}/files/{file_id}?with_result=1"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=acr_auth()), timeout=15) as r:
                d = json.loads(r.read())
        except urllib.error.HTTPError as e:
            err = e.read().decode()
            print(f"ACR poll error {e.code}: {err}")
            raise ValueError(f"Poll failed {e.code}: {err}")
        data = d.get("data", d)
        state = data.get("state")
        print(f"Poll state={state}")
        if state == 1:
            print(f"Result: {json.dumps(data)[:500]}")
            return data
        if isinstance(state, int) and state < 0:
            raise ValueError(f"ACRCloud error state={state}")
        time.sleep(3)
    raise TimeoutError("ACRCloud timed out after 90s")

def acr_delete(file_id):
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{acr_base()}/files/{file_id}",
            headers=acr_auth(), method="DELETE"), timeout=10)
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
    print(f"\n--- {req.artist} / {req.track} ---")
    preview = deezer_preview(req.isrc, req.artist, req.track)
    if not preview:
        raise HTTPException(404, "No Deezer preview found")
    file_id = None
    try:
        audio = fetch_audio(preview)
        print(f"Downloaded {len(audio)} bytes")
        result = acr_submit(audio, f"{req.artist} - {req.track}")
        file_id = str((result.get("data") or result).get("id", ""))
        if not file_id:
            raise ValueError(f"No file ID: {result}")
        print(f"File ID: {file_id}")
        data = acr_poll(file_id)
        ai = data.get("results", {}).get("ai_detection", [])
        print(f"AI detection: {ai}")
        if not ai:
            raise HTTPException(422, "No AI detection result — ensure AI Music Detection is enabled on container 31424")
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
