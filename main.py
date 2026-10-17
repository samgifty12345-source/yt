"""Faceless YouTube bot - PREVIEW build (no posting yet).

What it does, step by step:
  1. You give it a YouTube channel handle (e.g. @somechannel).
  2. It finds the channel's latest long-form video.
  3. Gemini WATCHES the video (opens the public YouTube link itself) and returns the
     scenes: what is said + what is on screen. Fallback: yt-dlp subtitles + scene detection.
  4. The AI rewrites the narration of every scene: new sentence structure,
     same meaning (facts, numbers, order of ideas all kept).
  5. Voice-over is generated for each scene (Fish Audio -> Microsoft free voice;
     Gemini voice only if USE_GEMINI_VOICE=1).
  6. Krea redraws each scene in YOUR animation style (your reference image).
     If the next scene looks the same as the previous one, the image is reused.
  7. Every image gets a slow zoom / pan that lasts exactly as long as that
     scene's voice-over, then everything is stitched into one video.

Output: data/out/preview.mp4 + data/out/script.txt (original vs rewritten, per
scene, so you can check the meaning really stayed the same).

Env vars:
  GEMINI_API_KEY      Google Gemini key (FREE tier works) - watch + rewrite + frame description
                      get one at aistudio.google.com -> Get API key
  ANTHROPIC_API_KEY   optional alternative to Gemini (paid). Gemini is used if both are set
  GEMINI_MODEL        optional, default gemini-2.5-flash (if Google removed it, the bot
                      automatically picks the newest Gemini Flash model)
  WATCH_WITH_GEMINI   optional, set to 0 to skip the Gemini 'watch' step
  KREA_API_KEY        Krea API token (krea.ai/settings/api-tokens)
  FISH_API_KEY        Fish Audio key (optional now - skipped automatically if it has no credit)
  FISH_VOICE_ID       optional, your Fish voice model id (default voice if empty)
  FISH_MODEL          optional, default s2-pro
  Voice order: Fish Audio -> ElevenLabs -> (Gemini voice, if USE_GEMINI_VOICE = True in the code) -> free Microsoft voice.
  If a voice fails part-way, ALL lines are redone with the next voice so the voice never changes.
  ELEVENLABS_API_KEY  ElevenLabs key (ELEVEN_VOICE_ID / ELEVEN_MODEL optional)
  GEMINI_VOICE        optional Gemini voice name (default Kore)
  GEMINI_TTS_MODEL    optional, default gemini-3.1-flash-tts-preview (auto-picks a TTS model if missing)
  EDGE_VOICE          free Microsoft voice (default en-US-AndrewNeural; others: en-US-BrianNeural,
                      en-US-ChristopherNeural, en-GB-RyanNeural, en-US-AriaNeural, en-US-JennyNeural)
  CLAUDE_MODEL        optional (only if using Anthropic), default claude-sonnet-5-5
  KREA_MODEL          optional, default krea/krea-2/medium
  STYLE_HINT          optional text added to every image prompt
                      (e.g. "flat 2D cartoon, thick outlines")
  STYLE_STRENGTH      optional 0-1, how strongly the reference style applies (0.8)
  PREVIEW_SECONDS     optional, how much of the source video to rebuild (90)
  OUT_W / OUT_H       optional output size, default 1280x720
  YOUTUBE_COOKIES_URL optional GitHub link to a cookies.txt (only used by the downloader fallback).
                      If the repo is PRIVATE also set GITHUB_TOKEN.
  YOUTUBE_COOKIES_B64 optional base64 of a YouTube cookies.txt (downloader fallback only)
  YOUTUBE_COOKIES_FILE optional path to a cookies.txt already on disk
  PROXY_URL           optional residential proxy (http://user:pass@host:port) for YouTube requests
  DASHBOARD_PASSWORD  optional
  PORT                optional, default 8080

Needs: pip install requests yt-dlp edge-tts   and ffmpeg + ffprobe installed.

Run the dashboard:   python faceless_bot.py
Or one-shot from CLI: python faceless_bot.py @channelhandle 90
"""
import os
import re
import sys
import html
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
EDGE_VOICE = os.environ.get("EDGE_VOICE", "en-US-AndrewNeural")
GEMINI_VOICE = os.environ.get("GEMINI_VOICE", "Kore")
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
# Edit these right here, no env vars needed:
USE_GEMINI_VOICE = False   # True = try Gemini's voice (tiny free daily quota) before the Microsoft voice
ELEVEN_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "") or os.environ.get("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "pNInz6obpgDQGcFmaJgB")   # "Adam" (free-plan premade voice)
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_multilingual_v2")
STYLE_HINT = os.environ.get("STYLE_HINT", "").strip()
STYLE_STRENGTH = float(os.environ.get("STYLE_STRENGTH", "0.8"))
PREVIEW_SECONDS = int(os.environ.get("PREVIEW_SECONDS", "90"))
OUT_W = int(os.environ.get("OUT_W", "1280"))
OUT_H = int(os.environ.get("OUT_H", "720"))
FPS = 30
SUB_LANGS = os.environ.get("SUB_LANGS", "en,en.*").split(",")
YOUTUBE_COOKIES_FILE = os.environ.get("YOUTUBE_COOKIES_FILE", "")
YOUTUBE_COOKIES_URL = os.environ.get("YOUTUBE_COOKIES_URL", "").strip()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
PROXY_URL = os.environ.get("PROXY_URL", "").strip()
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
    if PROXY_URL and "youtube.com" in url:
        kw.setdefault("proxies", {"http": PROXY_URL, "https": PROXY_URL})
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
    r = None
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
    print(f"yt-dlp update failed: {r.stderr[-200:] if r else ''}", flush=True)
    return False


def pip_install(pkg):
    for extra in ([], ["--break-system-packages"]):
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-U", "-q", pkg] + extra,
                               capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                return True
        except Exception:
            return False
    return False


def startup_update():
    try:
        update_ytdlp(False)
        pip_install("youtube-transcript-api")
        pip_install("edge-tts")
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


def ydl_opts(cookies=True, **extra):
    o = {"quiet": True, "no_warnings": True}
    if PROXY_URL:
        o["proxy"] = PROXY_URL
    cp = cookies_path() if cookies else ""
    if cp:
        o["extractor_args"] = {"youtubetab": {"skip": ["authcheck"]}}   # = --extractor-args youtubetab:skip=authcheck
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


def latest_via_rss(handle):
    """Fallback: the channel's public RSS feed (no login needed)."""
    page = http("GET", channel_url(handle).rsplit("/videos", 1)[0], timeout=30,
                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"})
    m = re.search(r'"(?:channelId|externalId)":"(UC[\w-]{22})"', page.text) or \
        re.search(r'channel_id=(UC[\w-]{22})', page.text)
    if not m:
        return None
    feed = http("GET", f"https://www.youtube.com/feeds/videos.xml?channel_id={m.group(1)}", timeout=30)
    e = re.search(r"<entry>.*?<yt:videoId>([^<]+)</yt:videoId>.*?<title>([^<]*)</title>", feed.text, re.S)
    if not e:
        return None
    return {"id": e.group(1), "title": html.unescape(e.group(2)),
            "url": f"https://www.youtube.com/watch?v={e.group(1)}"}


def latest_video(handle):
    last = None
    try:                                  # public RSS feed: fast and not blocked on servers
        v = latest_via_rss(handle)
        if v:
            return v
    except Exception as e:
        last = e
    import yt_dlp
    for use_cookies in (False, True):     # backup: the channel page
        try:
            with yt_dlp.YoutubeDL(ydl_opts(cookies=use_cookies, extract_flat=True, playlistend=3)) as y:
                info = y.extract_info(channel_url(handle), download=False)
            v = first_video(info or {})
            if v:
                return {"id": v["id"], "title": v.get("title") or v["id"],
                        "url": f"https://www.youtube.com/watch?v={v['id']}"}
        except Exception as e:
            last = e
    raise last or RuntimeError("couldn't find any videos on that channel")


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


def subtitles_via_transcript_api(video_id):
    """Second way to get captions: the youtube-transcript-api package (also needs a non-datacenter IP)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        raise RuntimeError("youtube-transcript-api is not installed")
    kw = {}
    if PROXY_URL:
        from youtube_transcript_api.proxies import GenericProxyConfig
        kw["proxy_config"] = GenericProxyConfig(http_url=PROXY_URL, https_url=PROXY_URL)
    langs = [l for l in SUB_LANGS if ".*" not in l] or ["en"]
    cues, prev = [], ""
    for sn in YouTubeTranscriptApi(**kw).fetch(video_id, languages=langs):
        txt = re.sub(r"\s+", " ", sn.text.replace("\n", " ")).strip()
        if txt and txt != prev:
            prev = txt
            cues.append((sn.start, sn.start + sn.duration, txt))
    if not cues:
        raise RuntimeError("no captions returned")
    return cues


def get_subtitles(video_url, tmp):
    try:
        return get_subtitles_ytdlp(video_url, tmp)
    except Exception as e1:
        vid = (urllib.parse.parse_qs(urllib.parse.urlparse(video_url).query).get("v") or [""])[0]
        status("📝 yt-dlp couldn't get the subtitles, trying youtube-transcript-api ...")
        try:
            return subtitles_via_transcript_api(vid)
        except Exception as e2:
            raise RuntimeError(f"couldn't get subtitles. yt-dlp: {str(e1)[:150]} | transcript-api: {str(e2)[:150]}")


def get_subtitles_ytdlp(video_url, tmp):
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
# LLM: rewrite narration + describe frames
# ---------------------------------------------------------------------------
_gemini_state = {"no_thinking": False, "picked": False}


def pick_gemini_model():
    """Newest plain 'gemini-X-flash' model this key can use."""
    r = http("GET", "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
             headers={"x-goog-api-key": GEMINI_API_KEY}, timeout=30)
    best = None
    for m in (r.json().get("models", []) if r.status_code == 200 else []):
        name = m.get("name", "").split("/")[-1]
        mm = re.match(r"^gemini-(\d+(?:\.\d+)?)-flash$", name)
        if mm and "generateContent" in m.get("supportedGenerationMethods", []):
            v = float(mm.group(1))
            if best is None or v > best[0]:
                best = (v, name)
    return best[1] if best else None


def gemini_call(system, parts, max_tokens=4000, json_mode=False, temperature=0.7, timeout=180):
    """One Gemini generateContent call. Handles a removed model name and thinking-config differences."""
    global GEMINI_MODEL
    r = None
    for _ in range(4):
        cfg = {"maxOutputTokens": max_tokens, "temperature": temperature}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        if "flash" in GEMINI_MODEL and not _gemini_state["no_thinking"]:
            cfg["thinkingConfig"] = {"thinkingBudget": 0}      # no hidden thinking tokens
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        r = http("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                 headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                 json=body, timeout=timeout, tries=5)
        if r.status_code == 404 and not _gemini_state["picked"]:
            _gemini_state["picked"] = True
            newm = pick_gemini_model()
            if newm and newm != GEMINI_MODEL:
                status(f"ℹ️ Gemini model {esc(GEMINI_MODEL)} isn't available, switching to {esc(newm)}.")
                GEMINI_MODEL = newm
                continue
        if r.status_code == 400 and "thinking" in r.text.lower() and not _gemini_state["no_thinking"]:
            _gemini_state["no_thinking"] = True
            continue
        break
    if r.status_code != 200:
        raise RuntimeError(f"Gemini API {r.status_code}: {r.text[:250]}")
    j = r.json()
    cands = j.get("candidates") or []
    if not cands:
        raise RuntimeError(f"Gemini returned nothing: {str(j.get('promptFeedback', j))[:200]}")
    out = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", [])).strip()
    if not out:
        raise RuntimeError(f"Gemini returned empty text (finish: {cands[0].get('finishReason')})")
    return out


def llm(system, text, max_tokens=4000, image_b64=None):
    """One text (optionally + one JPEG image) in, text out. Uses Gemini (free tier ok)
    if GEMINI_API_KEY is set, otherwise Anthropic."""
    if GEMINI_API_KEY:
        parts = [{"text": text}]
        if image_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
        return gemini_call(system, parts, max_tokens)
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


WATCH_PROMPT = """Watch this video from the start. Split the first {N} seconds into scenes: begin a new scene whenever
the picture clearly changes (new image, new shot, new diagram, new background). Each scene should last
3 to 10 seconds (split longer shots, merge very short ones).

For every scene return:
- "start" and "end": seconds from the start of the video (numbers)
- "narration": the words spoken during the scene, transcribed exactly, plain text ("" if nobody speaks)
- "visual": ONE image-generation prompt (max 70 words) that recreates what is on screen as an animated
  illustration: main subject(s), pose and position, camera angle, setting, key objects, mood, colors.
  Do NOT name an art style, include no readable text or lettering, and describe people generically.
- "same_as_previous": true only if the picture is essentially unchanged from the previous scene.

Return ONLY a JSON list, for example:
[{"start":0,"end":6.5,"narration":"...","visual":"...","same_as_previous":false}]"""


def _secs(v):
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).strip().rstrip("s")
    if ":" in v:
        out = 0.0
        for part in v.split(":"):
            out = out * 60 + float(part)
        return out
    return float(v)


def gemini_watch(video_url, seconds):
    """Let Gemini open the YouTube link itself and describe the first `seconds`.
    Returns (scenes[[start,end,text]], visuals[prompt], owner[image index per scene])."""
    parts = [{"file_data": {"file_uri": video_url},
              "video_metadata": {"start_offset": "0s", "end_offset": f"{int(seconds)}s"}},
             {"text": WATCH_PROMPT.replace("{N}", str(int(seconds)))}]
    raw = gemini_call("", parts, max_tokens=16000, json_mode=True, temperature=0.3, timeout=600)
    items = parse_json_list(raw)
    scenes, visuals, same = [], [], []
    for it in items:
        try:
            a, b = _secs(it["start"]), _secs(it["end"])
        except (KeyError, ValueError):
            continue
        if a >= seconds or b <= a:
            continue
        text = re.sub(r"\s+", " ", str(it.get("narration") or "")).strip()
        vis = str(it.get("visual") or "").strip()
        if not text and scenes:                        # silent stretch joins the previous scene
            scenes[-1][1] = b
            continue
        if not text:                                   # silent start: carry its timing into the next scene
            scenes.append([a, b, ""])
        else:
            scenes.append([a, b, text])
        visuals.append(vis)
        same.append(bool(it.get("same_as_previous")))
    if len(scenes) > 1 and not scenes[0][2]:
        scenes[1][0] = scenes[0][0]
        scenes.pop(0), visuals.pop(0), same.pop(0)
    keep = [i for i, sc in enumerate(scenes) if sc[2] and visuals[i]]
    if not keep:
        raise RuntimeError("Gemini didn't return usable scenes")
    scenes = [scenes[i] for i in keep]
    visuals = [visuals[i] for i in keep]
    same = [same[i] for i in keep]
    owner = []
    for i in range(len(scenes)):
        owner.append(owner[i - 1] if i > 0 and same[i] else i)
    return scenes, visuals, owner


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


DESCRIBE_TEXT_SYSTEM = """You write image-generation prompts.
You get the narration of one scene from an explainer video. Write ONE prompt (max 70 words) for an animated
illustration that shows what the narration is about: main subject(s), pose and position, setting, key objects,
mood and colors. No text or lettering in the image. Do NOT name the art style (a style reference image
supplies it). Do not name real people; use generic characters. Output ONLY the prompt."""


def describe_frame(frame_path, narration):
    if not frame_path:     # couldn't watch the video: draw from the narration alone
        return llm(DESCRIBE_TEXT_SYSTEM, f"Narration: {narration[:500]}", 300)
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
# Voice-over: Fish Audio -> (Gemini, optional) -> free Microsoft voice
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


def eleven_tts(text, dest):
    """ElevenLabs voice-over (mp3)."""
    r = http("POST", f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}?output_format=mp3_44100_128",
             headers={"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"},
             json={"text": text, "model_id": ELEVEN_MODEL}, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:200]}")
    with open(dest, "wb") as f:
        f.write(r.content)


def edge_tts_save(text, dest):
    """Free Microsoft voice (edge-tts). No key, no credit."""
    import asyncio
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts isn't installed (pip install edge-tts)")
    last = None
    for i in range(3):                    # the unofficial service sometimes hiccups
        try:
            asyncio.run(edge_tts.Communicate(text, EDGE_VOICE).save(dest))
            if os.path.isfile(dest) and os.path.getsize(dest) >= 500:
                return
            last = RuntimeError("edge-tts returned no audio")
        except Exception as e:
            last = e
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"edge-tts failed: {last}")


def pick_tts_model():
    r = http("GET", "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
             headers={"x-goog-api-key": GEMINI_API_KEY}, timeout=30)
    names = sorted(m.get("name", "").split("/")[-1] for m in (r.json().get("models", []) if r.status_code == 200 else [])
                   if "tts" in m.get("name", "") and "generateContent" in m.get("supportedGenerationMethods", []))
    return names[-1] if names else None


def gemini_tts(text, dest):
    """Human-sounding voice from Google (free tier has a tiny daily quota). Returns mp3."""
    global GEMINI_TTS_MODEL
    body = {"contents": [{"parts": [{"text": "Say in a calm, engaging narrator voice: " + text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_VOICE}}}}}
    r = None
    for _ in range(2):
        r = http("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent",
                 headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                 json=body, timeout=180, tries=6)
        if r.status_code == 404:
            m = pick_tts_model()
            if m and m != GEMINI_TTS_MODEL:
                GEMINI_TTS_MODEL = m
                continue
        break
    if r.status_code != 200:
        raise RuntimeError(f"Gemini voice {r.status_code}: {r.text[:200]}")
    parts = ((r.json().get("candidates") or [{}])[0].get("content") or {}).get("parts", [])
    blob = next((p.get("inlineData") or p.get("inline_data") for p in parts if p.get("inlineData") or p.get("inline_data")), None)
    if not blob or not blob.get("data"):
        raise RuntimeError("Gemini voice returned no audio")
    rate = re.search(r"rate=(\d+)", blob.get("mimeType") or blob.get("mime_type") or "")
    pcm = dest + ".pcm"
    with open(pcm, "wb") as f:
        f.write(base64.b64decode(blob["data"]))
    try:
        run_ffmpeg(["ffmpeg", "-y", "-f", "s16le", "-ar", rate.group(1) if rate else "24000", "-ac", "1",
                    "-i", pcm, "-c:a", "libmp3lame", "-q:a", "3", dest], "convert Gemini voice")
    finally:
        try:
            os.remove(pcm)
        except OSError:
            pass


def make_voiceovers(texts, audios):
    """Try the voices in order. The first that voices EVERY line wins. If a voice fails on the first
    line or runs out part-way, all lines are redone with the next voice (so the voice never changes
    halfway through the video)."""
    options = []
    if FISH_API_KEY:
        options.append(("Fish Audio", fish_tts, WORKERS))
    if ELEVEN_API_KEY:
        options.append(("ElevenLabs", eleven_tts, 2))
    if GEMINI_API_KEY and USE_GEMINI_VOICE:
        options.append(("Gemini voice (free)", gemini_tts, 1))
    options.append(("free Microsoft voice", edge_tts_save, WORKERS))
    for k, (name, fn, workers) in enumerate(options):
        last = k == len(options) - 1
        try:
            fn(texts[0], audios[0])
        except Exception as e:
            if last:
                raise
            why = "no API credit" if "402" in str(e) else str(e)[:100]
            status(f"⚠️ {name} unavailable ({esc(why)}). Trying the next voice ...")
            continue
        status(f"🎙️ Voice: {name}")
        try:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(lambda i: fn(texts[i], audios[i]), range(1, len(texts))))
        except Exception as e:
            if last:
                raise
            status(f"⚠️ {name} ran out mid-way ({esc(str(e)[:80])}). Redoing all lines with the next voice ...")
            continue
        return name


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
    missing = [n for n, v in (("KREA_API_KEY", KREA_API_KEY),) if not v]
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
    seconds = max(20, int(seconds or PREVIEW_SECONDS))
    work = tempfile.mkdtemp(prefix="fl_")
    try:
        status(f"🔎 Looking up the latest video on {esc(handle)} ...")
        vid = yt_call(latest_video, handle)
        status(f"📺 Latest video: {esc(vid['title'][:90])}")

        scenes = visuals = owner = frames = None
        if GEMINI_API_KEY and os.environ.get("WATCH_WITH_GEMINI", "1") != "0":
            status("👀 Asking Gemini to watch the video (it opens the YouTube link itself - no download needed) ...")
            try:
                scenes, visuals, owner = gemini_watch(vid["url"], seconds)
                n = len(scenes)
                frames = [None] * n
                unique = sorted(set(owner))
                status(f"🎬 Gemini watched it: {n} scenes, {len(unique)} different looks")
            except Exception as e:
                print(f"Gemini watch failed: {e}", flush=True)
                scenes = None
                status(f"⚠️ Gemini couldn't watch it ({esc(str(e)[:200])}). Trying the downloader instead ...")

        if scenes is None:
            fetch_cookies_from_github()
            status("📝 Getting subtitles ...")
            cues = yt_call(get_subtitles, vid["url"], work)
            status(f"📝 {len(cues)} subtitle lines")

            status(f"⬇️ Downloading the first {seconds}s to watch it ...")
            src = None
            try:
                src = yt_call(download_clip, vid["url"], work, seconds)
            except Exception as e:
                print(f"Video download failed: {e}", flush=True)
                status("⚠️ YouTube blocked the video download, so the bot can't watch it. "
                       "Building from the script only (images are drawn from the narration, not the original frames).")

            if src:
                total = min(duration(src), seconds)
                status("👀 Watching the video and finding scene changes ...")
                scenes = attach_text(build_scenes(detect_cuts(src, total), total), cues)
            else:
                total = min(float(seconds), max([b for _, b, _ in cues] + [0.0]))
                scenes = attach_text(build_scenes([], total), cues)
            if not scenes:
                raise RuntimeError("no spoken lines found in that stretch of the video")
            n = len(scenes)

            # one frame per scene; scenes that look the same as the previous one reuse its image
            frames, owner, hashes = [], [], []
            for i, (a, b, _) in enumerate(scenes):
                if not src:
                    frames.append(None)
                    owner.append(i)
                    continue
                fp = os.path.join(work, f"f{i:03d}.jpg")
                grab_frame(src, (a + b) / 2, fp)
                frames.append(fp)
                h = dhash(fp)
                same = (i > 0 and h is not None and hashes[i - 1] is not None
                        and bin(h ^ hashes[i - 1]).count("1") <= SAME_IMAGE_BITS)
                hashes.append(h)
                owner.append(owner[i - 1] if same else i)
            unique = sorted(set(owner))
            status(f"🎬 {n} scenes, {len(unique)} different looks"
                   + (" (reusing images when the video didn't change)" if src else ""))

        status("✍️ Rewriting the narration (same meaning, new structure) ...")
        new_text = rewrite_scenes(scenes, vid["title"])

        status("🎙️ Generating voice-over ...")
        audios = [os.path.join(work, f"a{i:03d}.mp3") for i in range(n)]
        voice_used = make_voiceovers(new_text, audios)

        status(f"🎨 Redrawing {len(unique)} frames in your style with Krea ...")
        ref = krea_reference_url()
        images, prompts = {}, {}

        def make_image(i):
            p = visuals[i] if visuals else describe_frame(frames[i], scenes[i][2])
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

        lines = [f"SOURCE: {vid['title']}  ({vid['url']})", f"VOICE: {voice_used}", ""]
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
<div class="card"><h2>YouTube cookies (only for the downloader fallback)</h2><div class="hint" style="margin:0 0 8px">@@COOKIES@@</div>
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
    if not KREA_API_KEY:
        warns.append("KREA_API_KEY is not set")
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
          "No cookies. Not needed while Gemini can watch the video.")
    rep = {"@@COOKIES@@": ck, "@@WARN@@": warn, "@@REF@@": ref, "@@FEED@@": feed, "@@VIDEO@@": video,
           "@@SECS@@": str(PREVIEW_SECONDS), "@@BTN@@": "Building..." if job_running else "Build preview",
           "@@RELOAD@@": "setTimeout(()=>location.reload(),4000);" if job_running else ""}
    page = PAGE
    for k, v in rep.items():
        page = page.replace(k, v)
    return page


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
