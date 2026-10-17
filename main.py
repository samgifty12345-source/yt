"""Faceless YouTube bot - PREVIEW build (no posting yet).

What it does, step by step:
  1. You give it a YouTube channel handle (e.g. @somechannel).
  2. It finds the channel's latest long-form video and downloads its subtitles.
  3. It watches the video (scene-change detection) and cuts it into scenes.
     Each scene = the stretch of video between two visual changes.
  4. the AI rewrites the narration of every scene: new sentence structure,
     same meaning (facts, numbers, order of ideas all kept).
  5. Fish Audio turns each rewritten scene into voice-over.
  6. For each scene, ONE frame is looked at by the AI (vision), which writes a
     prompt describing its composition. Krea then redraws that frame in YOUR
     animation style (your reference image). If the next scene looks the same
     as the previous one, the image is reused instead of regenerated.
  7. Every image gets a slow zoom / pan that lasts exactly as long as that
     scene's voice-over, then everything is stitched into one video.

Output: data/out/preview.mp4 + data/out/script.txt (original vs rewritten, per
scene, so you can check the meaning really stayed the same).

Env vars:
  GEMINI_API_KEY      Google Gemini key (FREE tier works) - rewrite + frame description
                      get one at aistudio.google.com -> Get API key
  ANTHROPIC_API_KEY   optional alternative to Gemini (paid). Gemini is used if both are set
  GEMINI_MODEL        optional, default gemini-2.5-flash
  KREA_API_KEY        Krea API token (krea.ai/settings/api-tokens)
  FISH_API_KEY        Fish Audio key
  FISH_VOICE_ID       optional, your Fish voice model id (default voice if empty)
  FISH_MODEL          optional, default s2-pro
  CLAUDE_MODEL        optional (only if using Anthropic), default claude-sonnet-5-5
  KREA_MODEL          optional, default krea/krea-2/medium
  STYLE_HINT          optional text added to every image prompt
                      (e.g. "flat 2D cartoon, thick outlines")
  STYLE_STRENGTH      optional 0-1, how strongly the reference style applies (0.8)
  PREVIEW_SECONDS     optional, how much of the source video to rebuild (90)
  OUT_W / OUT_H       optional output size, default 1280x720
  YOUTUBE_COOKIES_URL GitHub link to your cookies.txt (default: the one you gave me). The bot
                      downloads it at start and before every preview, so replacing the file
                      on GitHub refreshes it. If the repo is PRIVATE also set GITHUB_TOKEN
                      (token with read access to Contents).
  YOUTUBE_COOKIES_B64 base64 of a YouTube cookies.txt (needed on Railway: YouTube blocks
                      server IPs). Or upload cookies.txt on the dashboard instead.
  YOUTUBE_COOKIES_FILE optional path to a cookies.txt already on disk
  DASHBOARD_PASSWORD  optional
  PORT                optional, default 8080

Needs: pip install requests yt-dlp   and ffmpeg + ffprobe installed.

Run the dashboard:   python faceless_bot.py
Or one-shot from CLI: python faceless_bot.py @channelhandle 90
"""
import os
import re
import sys
import json
import math
import time
import hmac
import base64
import shutil
import tempfile
import threading
import subprocess
import mimetypes
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import requests

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
KREA_API_KEY = os.environ.get("KREA_API_KEY", "")
KREA_BASE = "https://api.krea.ai"
KREA_MODEL = os.environ.get("KREA_MODEL", "krea/krea-2/medium")
FISH_API_KEY = os.environ.get("FISH_API_KEY", "")
FISH_VOICE_ID = os.environ.get("FISH_VOICE_ID", "")
FISH_MODEL = os.environ.get("FISH_MODEL", "s2-pro")
STYLE_HINT = os.environ.get("STYLE_HINT", "").strip()
STYLE_STRENGTH = float(os.environ.get("STYLE_STRENGTH", "0.8"))
PREVIEW_SECONDS = int(os.environ.get("PREVIEW_SECONDS", "90"))
OUT_W = int(os.environ.get("OUT_W", "1280"))
OUT_H = int(os.environ.get("OUT_H", "720"))
FPS = 30
SUB_LANGS = os.environ.get("SUB_LANGS", "en,en.*").split(",")
YOUTUBE_COOKIES_FILE = os.environ.get("YOUTUBE_COOKIES_FILE", "")
YOUTUBE_COOKIES_URL = os.environ.get("YOUTUBE_COOKIES_URL",
                                     "https://github.com/samgifty12345-source/bedu/blob/main/cookies%20(1).txt").strip()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
FFMPEG_THREADS = os.environ.get("FFMPEG_THREADS", "2")

SCENE_THRESHOLD = 0.30   # lower = more scene cuts
MIN_SCENE = 3.0          # seconds, shorter scenes are merged
MAX_SCENE = 10.0         # seconds, longer scenes are split
SAME_IMAGE_BITS = 6      # frames this similar share one generated image
WORKERS = 3              # parallel API calls

DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(DATA_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)
REF_EXT = (".png", ".jpg", ".jpeg", ".webp")

# ---------------------------------------------------------------------------
# Logging / helpers
# ---------------------------------------------------------------------------
status_lock = threading.Lock()
status_feed = []


def status(msg):
    print(f"[status] {msg}", flush=True)
    with status_lock:
        status_feed.append(msg)
        if len(status_feed) > 80:
            status_feed.pop(0)


def esc(s):
    return (str(s) if s is not None else "").replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def http(method, url, tries=3, **kw):
    """requests with a few retries on rate limits / server errors."""
    kw.setdefault("timeout", 120)
    last = None
    for i in range(tries):
        try:
            r = requests.request(method, url, **kw)
            if r.status_code in (429, 500, 502, 503, 529) and i < tries - 1:
                time.sleep(12 * (i + 1) if r.status_code == 429 else 3 * (i + 1))
                continue
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"network error calling {url.split('?')[0]}: {last}")


def run_ffmpeg(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        if r.returncode < 0:
            raise RuntimeError(f"{what}: ffmpeg killed (out of memory? lower FFMPEG_THREADS / OUT_W)")
        tail = "\n".join([l for l in r.stderr.splitlines() if l.strip()][-6:])
        raise RuntimeError(f"{what}: ffmpeg failed: {tail}")
    return r


def duration(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def reference_path():
    for ext in REF_EXT:
        p = os.path.join(DATA_DIR, "reference" + ext)
        if os.path.isfile(p):
            return p
    return ""


# ---------------------------------------------------------------------------
# YouTube: latest video, subtitles, clip
# ---------------------------------------------------------------------------
COOKIES_SAVED = os.path.join(DATA_DIR, "youtube_cookies.txt")


def load_cookies_env():
    b64 = os.environ.get("YOUTUBE_COOKIES_B64", "").strip()
    if b64:
        try:
            with open(COOKIES_SAVED, "wb") as f:
                f.write(base64.b64decode(b64))
        except Exception as e:
            print(f"Could not decode YOUTUBE_COOKIES_B64: {e}", flush=True)


def fetch_cookies_from_github():
    """Download the cookies.txt from GitHub (blob link works). Keeps the old copy if it fails."""
    if not YOUTUBE_COOKIES_URL:
        return
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/(?:blob|raw)/([^/]+)/(.+)$", YOUTUBE_COOKIES_URL)
    if not m:
        status("⚠️ YOUTUBE_COOKIES_URL must be a github.com .../blob/<branch>/<file> link.")
        return
    owner, repo, branch, path = m.groups()
    path = urllib.parse.quote(urllib.parse.unquote(path))
    try:
        if GITHUB_TOKEN:
            r = http("GET", f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}",
                     headers={"Authorization": f"Bearer {GITHUB_TOKEN}",
                              "Accept": "application/vnd.github.raw+json"}, timeout=30)
        else:
            r = http("GET", f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}", timeout=30)
        if r.status_code != 200:
            hint = " (private repo? set GITHUB_TOKEN)" if r.status_code in (401, 403, 404) and not GITHUB_TOKEN else ""
            raise RuntimeError(f"HTTP {r.status_code}{hint}")
        if b"youtube.com" not in r.content:
            raise RuntimeError("that file doesn't look like a YouTube cookies.txt")
        with open(COOKIES_SAVED, "wb") as f:
            f.write(r.content)
        print("YouTube cookies loaded from GitHub", flush=True)
    except Exception as e:
        status(f"⚠️ Couldn't load cookies from GitHub: {esc(str(e)[:200])}"
               + (" - using the last saved copy." if cookies_path() else ""))


# --- keep yt-dlp up to date (YouTube changes break old versions, e.g. "The page needs to be reloaded") ---
ytdlp_ready = threading.Event()
_nightly_tried = [False]


def update_ytdlp(nightly=False):
    """pip-upgrade yt-dlp, then drop the old copy from memory so the next import is the new one."""
    cmd = [sys.executable, "-m", "pip", "install", "-U", "-q"] + (["--pre"] if nightly else []) + ["yt-dlp[default]"]
    for extra in ([], ["--break-system-packages"]):
        try:
            r = subprocess.run(cmd + extra, capture_output=True, text=True, timeout=300)
        except Exception as e:
            print(f"yt-dlp update error: {e}", flush=True)
            return False
        if r.returncode == 0:
            for m in [m for m in sys.modules if m == "yt_dlp" or m.startswith("yt_dlp.")]:
                del sys.modules[m]
            try:
                import yt_dlp
                print(f"yt-dlp is now {yt_dlp.version.__version__}", flush=True)
            except Exception:
                pass
            return True
    print(f"yt-dlp update failed: {r.stderr[-200:]}", flush=True)
    return False


def startup_update():
    try:
        update_ytdlp(False)
    finally:
        ytdlp_ready.set()


def yt_call(fn, *args):
    """Run a yt-dlp step; if YouTube says 'page needs to be reloaded', upgrade to the newest yt-dlp once and retry."""
    ytdlp_ready.wait(timeout=300)
    try:
        return fn(*args)
    except Exception as e:
        if "needs to be reloaded" in str(e) and not _nightly_tried[0]:
            _nightly_tried[0] = True
            status("🔄 YouTube changed something - upgrading yt-dlp and retrying ...")
            if update_ytdlp(True):
                return fn(*args)
        raise


def cookies_path():
    for p in (YOUTUBE_COOKIES_FILE, COOKIES_SAVED):
        if p and os.path.isfile(p):
            return p
    return ""


def ydl_opts(**extra):
    o = {"quiet": True, "no_warnings": True,
         "extractor_args": {"youtubetab": {"skip": ["authcheck"]}}}   # same as --extractor-args youtubetab:skip=authcheck
    cp = cookies_path()
    if cp:
        # yt-dlp rewrites the cookie file, so give it a private copy
        tmp_cp = os.path.join(tempfile.gettempdir(), "yt_cookies_copy.txt")
        shutil.copyfile(cp, tmp_cp)
        o["cookiefile"] = tmp_cp
    o.update(extra)
    return o


def channel_url(handle):
    h = handle.strip()
    if h.startswith("http"):
        if re.search(r"/(videos|streams|shorts)/?$", h):
            return h
        if re.search(r"youtube\.com/(@|channel/|c/|user/)", h):
            return h.rstrip("/") + "/videos"
        return h
    if not h.startswith("@"):
        h = "@" + h
    return f"https://www.youtube.com/{h}/videos"


def first_video(info):
    for e in info.get("entries") or []:
        if not e:
            continue
        if e.get("entries"):
            v = first_video(e)
            if v:
                return v
        elif e.get("id"):
            return e
    return None


def latest_video(handle):
    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts(extract_flat=True, playlistend=3)) as y:
        info = y.extract_info(channel_url(handle), download=False)
    v = first_video(info or {})
    if not v:
        raise RuntimeError("couldn't find any videos on that channel")
    return {"id": v["id"], "title": v.get("title") or v["id"],
            "url": f"https://www.youtube.com/watch?v={v['id']}"}


def parse_json3(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cues, prev = [], ""
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        txt = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in segs).replace("\n", " ")).strip()
        if not txt or txt == prev:
            continue
        prev = txt
        st = ev.get("tStartMs", 0) / 1000.0
        cues.append((st, st + ev.get("dDurationMs", 0) / 1000.0, txt))
    return cues


def get_subtitles(video_url, tmp):
    import yt_dlp
    opts = ydl_opts(skip_download=True, writesubtitles=True, writeautomaticsub=True,
                    subtitleslangs=SUB_LANGS, subtitlesformat="json3",
                    outtmpl=os.path.join(tmp, "subs.%(ext)s"))
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([video_url])
    files = [os.path.join(tmp, n) for n in os.listdir(tmp) if n.endswith(".json3")]
    if not files:
        raise RuntimeError("this video has no subtitles/captions to work from")
    files.sort(key=lambda p: (not p.endswith(".en.json3"), "orig" in p, p))
    return parse_json3(files[0])


def download_clip(video_url, tmp, seconds):
    import yt_dlp
    from yt_dlp.utils import download_range_func
    opts = ydl_opts(format="bv*[height<=480]/b[height<=480]/b",
                    outtmpl=os.path.join(tmp, "src.%(ext)s"),
                    download_ranges=download_range_func(None, [(0, seconds)]),
                    force_keyframes_at_cuts=False)
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([video_url])
    f = next((os.path.join(tmp, n) for n in os.listdir(tmp) if n.startswith("src.")), None)
    if not f:
        raise RuntimeError("video download failed")
    return f


# ---------------------------------------------------------------------------
# Watching the video: scenes, key frames, "is this frame new?"
# ---------------------------------------------------------------------------
def detect_cuts(video, limit):
    r = subprocess.run(["ffmpeg", "-i", video, "-t", str(limit), "-an",
                        "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=900)
    return sorted(float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr))


def build_scenes(cuts, total):
    pts = [0.0]
    for c in cuts:
        if c - pts[-1] >= MIN_SCENE and total - c >= 1.5:
            pts.append(c)
    pts.append(total)
    scenes = []
    for a, b in zip(pts, pts[1:]):
        n = max(1, math.ceil((b - a) / MAX_SCENE))
        step = (b - a) / n
        for i in range(n):
            scenes.append([a + i * step, a + (i + 1) * step])
    return scenes


def attach_text(scenes, cues):
    for s in scenes:
        s.append(" ".join(t for a, b, t in cues if s[0] <= (a + b) / 2 < s[1]).strip())
    merged = []
    for a, b, t in scenes:               # scenes with no speech join the previous one
        if not t and merged:
            merged[-1][1] = b
        else:
            merged.append([a, b, t])
    if len(merged) > 1 and not merged[0][2]:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [s for s in merged if s[2]]


def grab_frame(video, t, dest):
    run_ffmpeg(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                "-vf", "scale=768:-2", "-q:v", "3", dest], "grab frame")


def dhash(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=9:8:flags=area,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True, timeout=30)
    b = r.stdout
    if len(b) < 72:
        return None
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (1 if b[y * 9 + x] > b[y * 9 + x + 1] else 0)
    return bits


# ---------------------------------------------------------------------------
# Claude: rewrite narration + describe frames
# ---------------------------------------------------------------------------
def llm(system, text, max_tokens=4000, image_b64=None):
    """One text (optionally + one JPEG image) in, text out. Uses Gemini (free tier ok)
    if GEMINI_API_KEY is set, otherwise Anthropic."""
    if GEMINI_API_KEY:
        parts = [{"text": text}]
        if image_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
        cfg = {"maxOutputTokens": max_tokens, "temperature": 0.7}
        if "flash" in GEMINI_MODEL:
            cfg["thinkingConfig"] = {"thinkingBudget": 0}      # no hidden thinking tokens
        r = http("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                 headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                 json={"systemInstruction": {"parts": [{"text": system}]},
                       "contents": [{"role": "user", "parts": parts}], "generationConfig": cfg},
                 timeout=180, tries=5)
        if r.status_code != 200:
            raise RuntimeError(f"Gemini API {r.status_code}: {r.text[:200]}")
        j = r.json()
        cands = j.get("candidates") or []
        if not cands:
            raise RuntimeError(f"Gemini returned nothing: {str(j.get('promptFeedback', j))[:200]}")
        out = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", [])).strip()
        if not out:
            raise RuntimeError(f"Gemini returned empty text (finish: {cands[0].get('finishReason')})")
        return out
    content = [{"type": "text", "text": text}]
    if image_b64:
        content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}})
    r = http("POST", "https://api.anthropic.com/v1/messages",
             headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                      "content-type": "application/json"},
             json={"model": CLAUDE_MODEL, "max_tokens": max_tokens, "system": system,
                   "messages": [{"role": "user", "content": content}]}, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Claude API {r.status_code}: {r.text[:200]}")
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()


REWRITE_SYSTEM = """You rewrite spoken video narration.
You get a JSON list of {"id", "text"} items. Together they are ONE continuous narration, cut into
fragments (they come from auto-captions, may lack punctuation, and may start or end mid-sentence).

Return the same JSON list with every "text" rewritten:
- Change the sentence structure AND the wording of every line.
- Keep the exact meaning: every fact, number, name, claim and the order of ideas stay the same.
- Add nothing new and drop nothing.
- Keep each item's content inside its own item (voice-over timing depends on it) and keep its length
  within about 10% of the original word count.
- Read one after another the items must still flow naturally.
- Plain spoken language for text-to-speech: no emojis, no stage directions, no markdown.
Output ONLY the JSON list."""

DESCRIBE_SYSTEM = """You write image-generation prompts.
You see one frame from a video plus the narration spoken over that scene. Write ONE prompt (max 70 words)
that recreates the frame's composition as an animated illustration: the main subject(s), their pose and
position, the camera angle, the setting, key objects, mood and colors. If the frame shows on-screen text,
leave it out or turn it into a simple icon. Do NOT name the art style (a style reference image supplies it).
Do not name or describe real people's faces; use generic characters instead. Output ONLY the prompt."""


def parse_json_list(text):
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b < 0:
        raise ValueError("no JSON list in reply")
    return json.loads(text[a:b + 1])


def rewrite_scenes(scenes, title):
    new = {}
    for start in range(0, len(scenes), 30):
        chunk = [{"id": i, "text": scenes[i][2]} for i in range(start, min(start + 30, len(scenes)))]
        msg = f"Video title (context only): {title}\n\n{json.dumps(chunk, ensure_ascii=False)}"
        for attempt in range(2):
            try:
                for d in parse_json_list(llm(REWRITE_SYSTEM, msg, 8000)):
                    new[int(d["id"])] = str(d["text"]).strip()
                break
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                if attempt:
                    raise RuntimeError(f"couldn't parse the AI's rewrite: {e}")
    return [new.get(i) or scenes[i][2] for i in range(len(scenes))]


def describe_frame(frame_path, narration):
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return llm(DESCRIBE_SYSTEM, f"Narration for this scene: {narration[:400]}", 300, image_b64=b64)


# ---------------------------------------------------------------------------
# Krea: your style -> images
# ---------------------------------------------------------------------------
def krea_headers():
    return {"Authorization": f"Bearer {KREA_API_KEY}"}


_ref_url = {}


def krea_reference_url():
    p = reference_path()
    if not p:
        return ""
    key = (p, os.path.getmtime(p))
    if key not in _ref_url:
        with open(p, "rb") as f:
            r = http("POST", f"{KREA_BASE}/assets", headers=krea_headers(),
                     files={"file": (os.path.basename(p), f, mimetypes.guess_type(p)[0] or "image/png")},
                     data={"description": "style reference"})
        if r.status_code >= 300:
            raise RuntimeError(f"Krea asset upload failed: {r.status_code} {r.text[:200]}")
        _ref_url[key] = r.json()["image_url"]
    return _ref_url[key]


def krea_wait(job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = http("GET", f"{KREA_BASE}/jobs/{job_id}", headers=krea_headers(), timeout=30)
        if r.status_code >= 300:
            raise RuntimeError(f"Krea job check failed: {r.status_code} {r.text[:200]}")
        j = r.json()
        if j.get("status") == "completed":
            return j["result"]["urls"][0]
        if j.get("status") in ("failed", "cancelled"):
            raise RuntimeError(f"Krea job {j.get('status')}: {str(j)[:200]}")
        time.sleep(2)
    raise RuntimeError("Krea job timed out")


# Different Krea models name the size option differently; try them in order
# and remember the first one that is accepted.
_size_variant = [0]
SIZE_VARIANTS = [{"aspect_ratio": "16:9", "resolution": "1K"}, {"aspect_ratio": "16:9"},
                 {"width": 1344, "height": 768}, {}]


def krea_image(prompt, ref_url, dest):
    body = {"prompt": prompt}
    if ref_url:
        body["image_style_references"] = [{"url": ref_url, "strength": STYLE_STRENGTH}]
    err = ""
    for vi in range(_size_variant[0], len(SIZE_VARIANTS)):
        r = http("POST", f"{KREA_BASE}/generate/image/{KREA_MODEL}", headers=krea_headers(),
                 json={**body, **SIZE_VARIANTS[vi]}, timeout=60)
        if r.status_code in (400, 422):
            err = r.text[:200]
            continue
        if r.status_code >= 300:
            raise RuntimeError(f"Krea {r.status_code}: {r.text[:200]}")
        _size_variant[0] = vi
        url = krea_wait(r.json()["job_id"])
        img = http("GET", url, timeout=120)
        if img.status_code != 200:
            raise RuntimeError(f"couldn't download Krea image ({img.status_code})")
        with open(dest, "wb") as f:
            f.write(img.content)
        return
    raise RuntimeError(f"Krea rejected the request: {err}")


# ---------------------------------------------------------------------------
# Fish Audio: voice-over
# ---------------------------------------------------------------------------
def fish_tts(text, dest):
    body = {"text": text, "format": "mp3"}
    if FISH_VOICE_ID:
        body["reference_id"] = FISH_VOICE_ID
    r = http("POST", "https://api.fish.audio/v1/tts",
             headers={"Authorization": f"Bearer {FISH_API_KEY}", "Content-Type": "application/json",
                      "model": FISH_MODEL}, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Fish Audio {r.status_code}: {r.text[:200]}")
    with open(dest, "wb") as f:
        f.write(r.content)


# ---------------------------------------------------------------------------
# Building the video
# ---------------------------------------------------------------------------
def scene_clip(img, audio, idx, dst):
    """Still image + slow zoom/pan lasting exactly as long as the scene's audio."""
    d = duration(audio) + 0.25
    frames = max(2, int(round(d * FPS)))
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    mode = idx % 3
    if mode == 0:      # slow zoom in
        z, x, y = f"1+0.12*on/{frames}", cx, cy
    elif mode == 1:    # slow zoom out
        z, x, y = f"1.12-0.12*on/{frames}", cx, cy
    else:              # slow pan left to right
        z, x, y = "1.12", f"(iw-iw/zoom)*on/{frames}", cy
    fc = (f"[0:v]scale={OUT_W * 2}:{OUT_H * 2}:force_original_aspect_ratio=increase,"
          f"crop={OUT_W * 2}:{OUT_H * 2},setsar=1,"
          f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={OUT_W}x{OUT_H}:fps={FPS},format=yuv420p[v];"
          f"[1:a]aresample=44100,apad[a]")
    run_ffmpeg(["ffmpeg", "-y", "-i", img, "-i", audio, "-filter_complex", fc,
                "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-threads", FFMPEG_THREADS, "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
                "-t", f"{d:.3f}", "-movflags", "+faststart", dst], f"scene {idx + 1}")


def concat(parts, out_path, workdir):
    lst = os.path.join(workdir, "list.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write("file '" + p.replace("'", "'\\''") + "'\n")
    run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                "-movflags", "+faststart", out_path], "join scenes")


def check_setup():
    missing = [n for n, v in (("KREA_API_KEY", KREA_API_KEY), ("FISH_API_KEY", FISH_API_KEY)) if not v]
    if not (GEMINI_API_KEY or ANTHROPIC_API_KEY):
        missing.insert(0, "GEMINI_API_KEY (free at aistudio.google.com)")
    if missing:
        raise RuntimeError("missing env vars: " + ", ".join(missing))
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is not installed")
    if not reference_path():
        status("ℹ️ No style reference image uploaded - images will use Krea's default look.")


def build_preview(handle, seconds=None):
    check_setup()
    fetch_cookies_from_github()
    seconds = max(20, int(seconds or PREVIEW_SECONDS))
    work = tempfile.mkdtemp(prefix="fl_")
    try:
        status(f"🔎 Looking up the latest video on {esc(handle)} ...")
        vid = yt_call(latest_video, handle)
        status(f"📺 Latest video: {esc(vid['title'][:90])}")

        status("📝 Getting subtitles ...")
        cues = yt_call(get_subtitles, vid["url"], work)
        status(f"📝 {len(cues)} subtitle lines")

        status(f"⬇️ Downloading the first {seconds}s to watch it ...")
        src = yt_call(download_clip, vid["url"], work, seconds)
        total = min(duration(src), seconds)

        status("👀 Watching the video and finding scene changes ...")
        scenes = attach_text(build_scenes(detect_cuts(src, total), total), cues)
        if not scenes:
            raise RuntimeError("no spoken lines found in that stretch of the video")
        n = len(scenes)

        # one frame per scene; scenes that look the same as the previous one reuse its image
        frames, owner, hashes = [], [], []
        for i, (a, b, _) in enumerate(scenes):
            fp = os.path.join(work, f"f{i:03d}.jpg")
            grab_frame(src, (a + b) / 2, fp)
            frames.append(fp)
            h = dhash(fp)
            same = (i > 0 and h is not None and hashes[i - 1] is not None
                    and bin(h ^ hashes[i - 1]).count("1") <= SAME_IMAGE_BITS)
            hashes.append(h)
            owner.append(owner[i - 1] if same else i)
        unique = sorted(set(owner))
        status(f"🎬 {n} scenes, {len(unique)} different looks (reusing images when the video didn't change)")

        status("✍️ Rewriting the narration (same meaning, new structure) ...")
        new_text = rewrite_scenes(scenes, vid["title"])

        status("🎙️ Generating voice-over with Fish Audio ...")
        audios = [os.path.join(work, f"a{i:03d}.mp3") for i in range(n)]
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(lambda i: fish_tts(new_text[i], audios[i]), range(n)))

        status(f"🎨 Redrawing {len(unique)} frames in your style with Krea ...")
        ref = krea_reference_url()
        images, prompts = {}, {}

        def make_image(i):
            p = describe_frame(frames[i], scenes[i][2])
            if STYLE_HINT:
                p += ". " + STYLE_HINT
            dest = os.path.join(work, f"img{i:03d}.png")
            krea_image(p, ref, dest)
            images[i], prompts[i] = dest, p

        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(make_image, unique))

        status("🎞️ Animating scenes and joining them ...")
        parts = []
        for i in range(n):
            dst = os.path.join(work, f"s{i:03d}.mp4")
            scene_clip(images[owner[i]], audios[i], i, dst)
            parts.append(dst)
        final = os.path.join(work, "final.mp4")
        concat(parts, final, work)
        shutil.move(final, os.path.join(OUT_DIR, "preview.mp4"))

        lines = [f"SOURCE: {vid['title']}  ({vid['url']})", ""]
        for i, (a, b, t) in enumerate(scenes):
            lines += [f"--- Scene {i + 1}  [{a:.1f}s-{b:.1f}s]  image from scene {owner[i] + 1}",
                      f"ORIGINAL : {t}", f"REWRITTEN: {new_text[i]}",
                      f"IMAGE PROMPT: {prompts.get(owner[i], '')}", ""]
        with open(os.path.join(OUT_DIR, "script.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        mb = os.path.getsize(os.path.join(OUT_DIR, "preview.mp4")) / 1_000_000
        status(f'✅ Preview ready ({mb:.1f} MB): <a href="/out/preview.mp4" target="_blank">watch</a> '
               f'&middot; <a href="/out/script.txt" target="_blank">original vs rewritten script</a>')
        return os.path.join(OUT_DIR, "preview.mp4")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------
job_lock = threading.Lock()
job_running = False


def run_job(handle, seconds):
    global job_running
    if not job_lock.acquire(blocking=False):
        status("⏳ A preview is already being built.")
        return
    job_running = True
    try:
        build_preview(handle, seconds)
    except Exception as e:
        print(f"Job failed: {type(e).__name__}: {e}", flush=True)
        if "Sign in to confirm" in str(e):
            have = "Your cookies were rejected or expired - export fresh ones" if cookies_path() else "Upload a YouTube cookies.txt on the dashboard (or set YOUTUBE_COOKIES_B64)"
            status(f"❌ YouTube thinks the server is a bot. {have}.")
        else:
            status(f"❌ {esc(str(e)[:400])}")
    finally:
        job_running = False
        job_lock.release()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Faceless Bot</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
:root{--bg:#0b0b10;--panel:#14141c;--p2:#1b1b26;--border:#26263a;--text:#eaeaf2;--muted:#8a8aa0;--a:#ff3b5c;--a2:#7c5cff}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Inter,Roboto,sans-serif;background:var(--bg);color:var(--text);padding:28px 18px 60px}
.wrap{max-width:900px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);font-size:13px;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:20px;margin-bottom:16px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 12px}
label{display:block;font-size:12.5px;color:var(--muted);margin:10px 0 6px}
input[type=text],input[type=number]{background:var(--p2);border:1px solid var(--border);border-radius:8px;padding:9px 10px;color:var(--text);font:inherit;width:100%}
button{background:linear-gradient(135deg,var(--a),var(--a2));color:#fff;border:0;padding:11px 16px;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;margin-top:14px}
.feed{display:flex;flex-direction:column;gap:7px;max-height:360px;overflow-y:auto}
.fi{background:#08080d;border:1px solid var(--border);border-radius:10px;padding:9px 12px;font-size:13px;line-height:1.5}
a{color:var(--a2);text-decoration:none}.hint{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.5}
.warn{border-color:rgba(255,59,92,.4);color:#ffb3c0;font-size:13px}video{width:100%;border-radius:12px;background:#000}
img.ref{max-height:140px;border-radius:10px;border:1px solid var(--border);display:block;margin-bottom:8px}
</style></head><body><div class="wrap">
<h1>Faceless Bot - preview</h1><div class="sub">Channel in, rebuilt video out. Nothing is posted.</div>
@@WARN@@
<div class="card"><h2>1. Your animation style</h2>@@REF@@
<input type="file" accept="image/png,image/jpeg,image/webp" onchange="up(this)"> <span id="upmsg" class="hint"></span>
<div class="hint">Every redrawn frame copies the look of this image.</div></div>
<div class="card"><h2>YouTube cookies</h2><div class="hint" style="margin:0 0 8px">@@COOKIES@@</div>
<input type="file" accept=".txt" onchange="upc(this)"> <span id="upc" class="hint"></span></div>
<div class="card"><h2>2. Build a preview</h2>
<form method="POST" action="/run"><label>YouTube channel handle or link</label>
<input type="text" name="handle" placeholder="@channelname" required>
<label>Seconds of the video to rebuild</label><input type="number" name="seconds" min="20" max="600" value="@@SECS@@">
<button type="submit">@@BTN@@</button></form>
<div class="hint">Short previews are cheap: about one Krea image per scene change.</div></div>
<div class="card"><h2>Activity</h2><div class="feed">@@FEED@@</div></div>
@@VIDEO@@
</div><script>
async function up(i){const f=i.files[0];if(!f)return;document.getElementById('upmsg').textContent='Uploading...';
const r=await fetch('/upload-ref?name='+encodeURIComponent(f.name),{method:'PUT',body:f});
if(!r.ok){document.getElementById('upmsg').textContent='Failed: '+await r.text();return}location.reload()}
async function upc(i){const f=i.files[0];if(!f)return;const r=await fetch('/upload-cookies',{method:'PUT',body:f});
if(!r.ok){document.getElementById('upc').textContent='Failed: '+await r.text();return}location.reload()}
@@RELOAD@@
</script></body></html>"""


def render_page():
    warns = []
    if not (GEMINI_API_KEY or ANTHROPIC_API_KEY):
        warns.append("GEMINI_API_KEY is not set (free key at aistudio.google.com)")
    for name, v in (("KREA_API_KEY", KREA_API_KEY), ("FISH_API_KEY", FISH_API_KEY)):
        if not v:
            warns.append(f"{name} is not set")
    warn = "".join(f'<div class="card warn">&#9888; {esc(w)}</div>' for w in warns)
    with status_lock:
        items = list(status_feed[-30:])
    feed = "\n".join(f'<div class="fi">{i}</div>' for i in reversed(items)) or '<div class="hint">Nothing yet.</div>'
    ref = f'<img class="ref" src="/ref?t={int(time.time())}">' if reference_path() else \
        '<div class="hint" style="margin:0 0 8px">No reference image yet.</div>'
    pv = os.path.join(OUT_DIR, "preview.mp4")
    video = ""
    if os.path.isfile(pv):
        video = (f'<div class="card"><h2>Latest preview</h2><video controls src="/out/preview.mp4?t={int(os.path.getmtime(pv))}">'
                 f'</video><div class="hint"><a href="/out/script.txt" target="_blank">Original vs rewritten script</a></div></div>')
    ck = ("Cookies loaded. If YouTube still blocks you, upload fresh ones." if cookies_path() else
          "No cookies yet. YouTube blocks server IPs, so upload a cookies.txt exported from a logged-in browser.")
    rep = {"@@COOKIES@@": ck, "@@WARN@@": warn, "@@REF@@": ref, "@@FEED@@": feed, "@@VIDEO@@": video,
           "@@SECS@@": str(PREVIEW_SECONDS), "@@BTN@@": "Building..." if job_running else "Build preview",
           "@@RELOAD@@": "setTimeout(()=>location.reload(),4000);" if job_running else ""}
    html = PAGE
    for k, v in rep.items():
        html = html.replace(k, v)
    return html


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _authorized(self):
        if not DASHBOARD_PASSWORD:
            return True
        h = self.headers.get("Authorization", "")
        if h.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(h[6:]).decode().partition(":")
                return hmac.compare_digest(pw, DASHBOARD_PASSWORD)
            except Exception:
                return False
        return False

    def _deny(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Faceless Bot"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path):
        if not os.path.isfile(path):
            return self._send(404, b"not found")
        size = os.path.getsize(path)
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            s, _, e = rng[6:].partition("-")
            try:
                start = int(s) if s else max(0, size - int(e))
                end = min(int(e) if (e and s) else size - 1, size - 1)
                code = 206
            except ValueError:
                start, end, code = 0, size - 1, 200
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".txt"):
            ctype = "text/plain; charset=utf-8"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1024 * 1024, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/healthz":
            return self._send(200, b"ok")
        if not self._authorized():
            return self._deny()
        if path == "/":
            return self._send(200, render_page().encode(), "text/html; charset=utf-8")
        if path in ("/out/preview.mp4", "/out/script.txt"):
            return self._serve_file(os.path.join(OUT_DIR, os.path.basename(path)))
        if path == "/ref":
            return self._serve_file(reference_path())
        self._send(404, b"not found")

    def do_PUT(self):
        if not self._authorized():
            return self._deny()
        u = urllib.parse.urlparse(self.path)
        if u.path == "/upload-cookies":
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 2 * 1024 * 1024:
                return self._send(413, b"cookies file too large")
            data = self.rfile.read(length)
            if b"youtube.com" not in data:
                return self._send(400, b"this doesn't look like a YouTube cookies.txt")
            with open(COOKIES_SAVED, "wb") as f:
                f.write(data)
            status("🍪 YouTube cookies saved.")
            return self._send(200, b"ok")
        if u.path != "/upload-ref":
            return self._send(404, b"not found")
        name = (urllib.parse.parse_qs(u.query).get("name") or [""])[0].lower()
        ext = os.path.splitext(name)[1]
        if ext not in REF_EXT:
            return self._send(400, b"use a png, jpg or webp image")
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 20 * 1024 * 1024:
            return self._send(413, b"image too large (max 20 MB)")
        data = self.rfile.read(length)
        for e in REF_EXT:
            try:
                os.remove(os.path.join(DATA_DIR, "reference" + e))
            except OSError:
                pass
        with open(os.path.join(DATA_DIR, "reference" + ext), "wb") as f:
            f.write(data)
        status("🖼️ Style reference image updated.")
        self._send(200, b"ok")

    def do_POST(self):
        if not self._authorized():
            return self._deny()
        if urllib.parse.urlparse(self.path).path != "/run":
            return self._send(404, b"not found")
        length = int(self.headers.get("Content-Length", 0))
        f = urllib.parse.parse_qs(self.rfile.read(length).decode() if length else "")
        handle = (f.get("handle") or [""])[0].strip()
        try:
            seconds = int((f.get("seconds") or [str(PREVIEW_SECONDS)])[0])
        except ValueError:
            seconds = PREVIEW_SECONDS
        if handle:
            status(f"▶️ Preview started for {esc(handle)}.")
            threading.Thread(target=run_job, args=(handle, seconds), daemon=True).start()
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    if len(sys.argv) > 1:                       # CLI mode
        load_cookies_env()
        startup_update()
        out = build_preview(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
        print("Done:", out)
        return
    port = int(os.environ.get("PORT", 8080))
    load_cookies_env()
    threading.Thread(target=startup_update, daemon=True).start()
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever, daemon=True).start()
    print(f"Faceless bot dashboard on port {port}", flush=True)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
