"""Faceless YouTube bot - v3 (Groq text, key pools, split-in-two, review gate, new dashboard).

What it does, step by step:
  1. You give it a channel handle (latest video) OR a direct link to ONE video.
  2. Gemini WATCHES the video (opens the public YouTube link itself) and returns the scenes. This is the only
     step that needs Gemini. Fallback: yt-dlp subtitles + scene detection.
  3. GROQ rewrites the narration (same meaning, new wording) and writes the title / description / tags, so your
     Gemini credits are only spent on watching. If Groq fails, Gemini/Anthropic can take over (a setting).
  4. Voice-over (Fish Audio -> ElevenLabs -> Microsoft free voice). If a backup voice had to be used, the video is
     flagged "needs review": it will NOT auto-post until you watch it and press Approve.
  5. The source video is downloaded once (Vid Kraken -> yt-dlp) into the VIDEO BANK, a real frame is cut per scene and
     Krea redraws it with TWO images (your style + that frame). Cheapest Krea model first, up to the medium one.
  6. Optional: split the video in two halves -> two separate videos (Part 1 / Part 2).
  7. Thumbnail redrawn from the source thumbnail, then optional posting to YouTube through Zernio (not made for kids).

KEYS: every provider can have several keys. Keys saved on the website (Keys tab) AND Railway env vars
(GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3 ...) are all pooled; when one fails (invalid, out of credit, rate
limited) the next is tried automatically. Paste a replacement on the Keys tab - no redeploy needed.

KILL SWITCH: stops everything and keeps the bot PAUSED (even after a redeploy) until you press Resume.

BANKS (data folder - survive a redeploy only with a Railway Volume mounted at /data):
  data/bank/videos/   source videos        data/jobs/<video>/  scenes, voice, frames, images, thumbnail, final video

Env vars (everything also editable on the dashboard, which wins over env):
  GROQ_API_KEY(+_2,_3..)  GROQ_MODELS  GROQ_VISION_MODEL  USE_GROQ  TEXT_FALLBACK
  GEMINI_API_KEY(+_2..)  ANTHROPIC_API_KEY  KREA_API_KEY  KREA_MODEL  KREA_LADDER  USE_CHEAP_KREA  FRAME_REFS
  VIDKRAKEN_API_KEY(+_2..)  ZERNIO_API_KEY  ZERNIO_ACCOUNT_ID  AUTO_POST  POST_VISIBILITY  MARK_SYNTHETIC
  SPLIT_IN_TWO  REVIEW_ON_FALLBACK  FISH_API_KEY  ELEVENLABS_API_KEY  CF_IMAGE_URL  CF_IMAGE_KEY  DASHBOARD_PASSWORD
  BACKUP_REPO / GITHUB_TOKEN / DATA_DIR / PROXY_URL / YOUTUBE_COOKIES_*  PORT

Needs: pip install requests yt-dlp edge-tts   and ffmpeg + ffprobe installed.
Run:  python faceless_bot.py      (CLI: python faceless_bot.py @channel 90   or a video link)
"""
import os
import re
import sys
import html
import json
import math
import time
import hmac
import secrets
import hashlib
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


def _envbool(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Settings (defaults come from env vars; the dashboard Settings tab overrides them)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODELS = os.environ.get("GROQ_MODELS", "llama-3.3-70b-versatile,openai/gpt-oss-120b,llama-3.1-8b-instant")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
USE_GROQ = _envbool("USE_GROQ", "1")
TEXT_FALLBACK = _envbool("TEXT_FALLBACK", "1")
SPLIT_IN_TWO = _envbool("SPLIT_IN_TWO", "0")
REVIEW_ON_FALLBACK = _envbool("REVIEW_ON_FALLBACK", "1")
KREA_API_KEY = os.environ.get("KREA_API_KEY", "")
KREA_BASE = "https://api.krea.ai"
KREA_MODEL = os.environ.get("KREA_MODEL", "krea/krea-2/medium")
KREA_LADDER = os.environ.get("KREA_LADDER", "krea/krea-1,bfl/flux-1-dev,krea/krea-2/medium")
USE_CHEAP_KREA = _envbool("USE_CHEAP_KREA", "1")
FRAME_REFS = _envbool("FRAME_REFS", "1")
VIDKRAKEN_API_KEY = os.environ.get("VIDKRAKEN_API_KEY", "")
ZERNIO_API_KEY = os.environ.get("ZERNIO_API_KEY", "")
ZERNIO_ACCOUNT_ID = os.environ.get("ZERNIO_ACCOUNT_ID", "")
AUTO_POST = _envbool("AUTO_POST", "0")
POST_VISIBILITY = os.environ.get("POST_VISIBILITY", "private")
MARK_SYNTHETIC = _envbool("MARK_SYNTHETIC", "0")
FISH_API_KEY = os.environ.get("FISH_API_KEY", "")
FISH_VOICE_ID = os.environ.get("FISH_VOICE_ID", "")
FISH_MODEL = os.environ.get("FISH_MODEL", "s2-pro")
EDGE_VOICE = os.environ.get("EDGE_VOICE", "en-US-AndrewNeural")
GEMINI_VOICE = os.environ.get("GEMINI_VOICE", "Kore")
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
USE_GEMINI_VOICE = os.environ.get("USE_GEMINI_VOICE", "0").lower() in ("1", "true", "yes")   # tiny free daily quota
ELEVEN_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "") or os.environ.get("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "auq43ws1oslv0tO4BDa7")   # your chosen voice (change it in Settings)
ELEVEN_MODEL = os.environ.get("ELEVEN_MODEL", "eleven_multilingual_v2")
# Free image fallback = your Cloudflare Worker.
CF_IMAGE_URL = os.environ.get("CF_IMAGE_URL", "") or "https://icon.larpwithjosh.workers.dev/"
CF_IMAGE_KEY = os.environ.get("CF_IMAGE_KEY", "") or "12345678"   # must match the worker's API_KEY secret
# Optional fallback for the Krea login if you have no volume and no backup repo (the dashboard shows it).
KREA_OAUTH_BLOB = os.environ.get("KREA_OAUTH_BLOB", "") or ""
CF_STYLE = "flat 2D cartoon illustration, thick outlines, simple shapes, vibrant colors, no text"
STYLE_HINT = os.environ.get("STYLE_HINT", "").strip()
STYLE_STRENGTH = float(os.environ.get("STYLE_STRENGTH", "0.8"))
PREVIEW_SECONDS = int(os.environ.get("PREVIEW_SECONDS", "90"))
OUT_W = int(os.environ.get("OUT_W", "1280"))
OUT_H = int(os.environ.get("OUT_H", "720"))
FPS = 30
WHOLE_MAX = int(os.environ.get("WHOLE_MAX_SECONDS", "1800"))   # longest video analysed with "whole video" (30 min)
SUB_LANGS = os.environ.get("SUB_LANGS", "en,en.*").split(",")
YOUTUBE_COOKIES_FILE = os.environ.get("YOUTUBE_COOKIES_FILE", "")
YOUTUBE_COOKIES_URL = os.environ.get("YOUTUBE_COOKIES_URL", "").strip()
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
PROXY_URL = os.environ.get("PROXY_URL", "").strip()
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
FFMPEG_THREADS = os.environ.get("FFMPEG_THREADS", "2")

BACKUP_REPO = os.environ.get("BACKUP_REPO", "").strip().strip("/")
BACKUP_BRANCH = os.environ.get("BACKUP_BRANCH", "main").strip() or "main"
BACKUP_TOKEN = os.environ.get("BACKUP_TOKEN", "").strip() or GITHUB_TOKEN
BACKUP_PATH = "faceless-bot-data"

SCENE_THRESHOLD = 0.30   # lower = more scene cuts
MIN_SCENE = 3.0          # seconds, shorter scenes are merged
MAX_SCENE = 10.0         # seconds, longer scenes are split
SAME_IMAGE_BITS = 6      # frames this similar share one generated image
WORKERS = 3              # parallel API calls

# Data folder: explicit DATA_DIR > Railway volume > ./data
DATA_DIR = (os.environ.get("DATA_DIR") or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
OUT_DIR = os.path.join(DATA_DIR, "out")
BANK_DIR = os.path.join(DATA_DIR, "bank")
VBANK_DIR = os.path.join(BANK_DIR, "videos")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VBANK_DIR, exist_ok=True)
REF_EXT = (".png", ".jpg", ".jpeg", ".webp")

# Fields editable on the dashboard (saved in data/settings.json). (name, label, kind)
SETTING_FIELDS = [
    ("USE_GROQ", "Use Groq for rewriting, titles and frame descriptions (saves Gemini credits)", "bool"),
    ("GROQ_MODELS", "Groq text models, in order (comma separated)", "text"),
    ("GROQ_VISION_MODEL", "Groq vision model (describes frames / thumbnails)", "text"),
    ("TEXT_FALLBACK", "If Groq fails, fall back to Gemini / Anthropic for text", "bool"),
    ("USE_CHEAP_KREA", "Try the cheapest Krea models first (falls back up to the medium one)", "bool"),
    ("KREA_LADDER", "Krea models, cheapest first (comma separated, the medium one is always last)", "text"),
    ("KREA_MODEL", "Krea medium model (last resort)", "text"),
    ("FRAME_REFS", "Send the real video frame to Krea as a 2nd reference image", "bool"),
    ("STYLE_HINT", "Style hint added to every image prompt", "text"),
    ("STYLE_STRENGTH", "Style strength (0 to 1)", "float"),
    ("CF_IMAGE_URL", "Cloudflare image worker URL (free fallback)", "text"),
    ("CF_IMAGE_KEY", "Cloudflare worker key", "secret"),
    ("SPLIT_IN_TWO", "Split every video into two halves (two separate videos)", "bool"),
    ("PREVIEW_SECONDS", "Default seconds to rebuild", "int"),
    ("ZERNIO_ACCOUNT_ID", "Zernio YouTube account id (blank = first YouTube account)", "text"),
    ("POST_VISIBILITY", "YouTube visibility: private / unlisted / public", "text"),
    ("AUTO_POST", "Post to YouTube automatically when a build finishes", "bool"),
    ("REVIEW_ON_FALLBACK", "Never auto-post when a backup voice was used (needs my approval)", "bool"),
    ("MARK_SYNTHETIC", "Tell YouTube the video contains AI / synthetic content", "bool"),
    ("FISH_VOICE_ID", "Fish voice id (optional)", "text"),
    ("ELEVEN_VOICE_ID", "ElevenLabs voice id", "text"),
    ("ELEVEN_MODEL", "ElevenLabs model", "text"),
    ("EDGE_VOICE", "Free Microsoft voice", "text"),
    ("USE_GEMINI_VOICE", "Try the Gemini voice before the free Microsoft voice", "bool"),
]
SETTING_GROUPS = [
    ("Text AI (Groq)", ["USE_GROQ", "GROQ_MODELS", "GROQ_VISION_MODEL", "TEXT_FALLBACK"]),
    ("Images (Krea)", ["USE_CHEAP_KREA", "KREA_LADDER", "KREA_MODEL", "FRAME_REFS", "STYLE_HINT", "STYLE_STRENGTH",
                       "CF_IMAGE_URL", "CF_IMAGE_KEY"]),
    ("Build", ["SPLIT_IN_TWO", "PREVIEW_SECONDS"]),
    ("Posting (Zernio)", ["AUTO_POST", "POST_VISIBILITY", "REVIEW_ON_FALLBACK", "MARK_SYNTHETIC", "ZERNIO_ACCOUNT_ID"]),
    ("Voice", ["FISH_VOICE_ID", "ELEVEN_VOICE_ID", "ELEVEN_MODEL", "EDGE_VOICE", "USE_GEMINI_VOICE"]),
]
ENV_DEFAULTS = {k: globals()[k] for k, _, _ in SETTING_FIELDS}
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

# ---------------------------------------------------------------------------
# KILL SWITCH
# ---------------------------------------------------------------------------
class Killed(BaseException):
    """Raised inside the pipeline when the kill switch is pressed (BaseException so no 'except Exception' eats it)."""


class OutOfCredit(RuntimeError):
    pass


class ModelRejected(RuntimeError):
    pass


kill_event = threading.Event()
PAUSED = False
_procs = set()
_procs_lock = threading.Lock()


def check_kill():
    if kill_event.is_set():
        raise Killed()


def sp_run(cmd, timeout=1800, text=True):
    """subprocess.run that the kill switch can terminate."""
    check_kill()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text)
    with _procs_lock:
        _procs.add(p)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        raise
    finally:
        with _procs_lock:
            _procs.discard(p)
    check_kill()
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def kill_everything():
    """Pause the whole bot: stop the running build/post, kill ffmpeg, refuse new work until Resume."""
    global PAUSED
    PAUSED = True
    kill_event.set()
    with _procs_lock:
        for p in list(_procs):
            try:
                p.kill()
            except Exception:
                pass
    try:
        settings_update({"PAUSED": "1"})
    except Exception as e:
        print(f"couldn't save the paused state: {e}", flush=True)
    status("⛔ KILL SWITCH pressed: everything is stopped and the bot is PAUSED. What it already made is saved below.")


def resume_bot():
    global PAUSED
    PAUSED = False
    kill_event.clear()
    try:
        settings_update({"PAUSED": None})
    except Exception as e:
        print(f"couldn't save the resumed state: {e}", flush=True)
    status("▶️ Bot resumed. You can build again.")


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
        check_kill()
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
    r = sp_run(cmd, timeout=1800)
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


def dir_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1_000_000


# ---------------------------------------------------------------------------
# API KEY POOL: several keys per provider (saved on the website + Railway env vars), tried in turn
# ---------------------------------------------------------------------------
class KeyFail(RuntimeError):
    """A key was refused (invalid / out of credit / rate-limited): the next key gets tried."""
    def __init__(self, msg, kind="invalid", retry=0):
        super().__init__(msg)
        self.kind = kind          # invalid | credit | rate
        self.retry = retry


KEY_PROVIDERS = [
    ("GROQ_API_KEY", "Groq", "Rewrites the narration, writes titles, describes frames (replaces Gemini for text)"),
    ("GEMINI_API_KEY", "Gemini", "Watches the video (the only step that needs it)"),
    ("ANTHROPIC_API_KEY", "Anthropic", "Optional text backup"),
    ("KREA_API_KEY", "Krea API", "Image generation through the API balance (your web credits use Connect Krea)"),
    ("VIDKRAKEN_API_KEY", "Vid Kraken", "Downloads the source video once (about 10 per day per key)"),
    ("ZERNIO_API_KEY", "Zernio", "Posts to YouTube"),
    ("FISH_API_KEY", "Fish Audio", "Voice-over"),
    ("ELEVEN_API_KEY", "ElevenLabs", "Voice-over"),
]
KEY_NAMES = [k for k, _, _ in KEY_PROVIDERS]
KEY_ENV_ALIASES = {"ELEVEN_API_KEY": ["ELEVENLABS_API_KEY", "ELEVEN_API_KEY"]}
KEY_COOLDOWN = {"invalid": 43200, "credit": 10800, "rate": 25}
_key_state = {}
_key_lock = threading.RLock()
_tl = threading.local()        # per-thread "current key" for providers whose jobs must stay on one key


def _kid(name, key):
    return name + ":" + hashlib.sha1(key.encode()).hexdigest()[:10]


def mask_key(key):
    return ("····" + key[-4:]) if len(key) > 6 else "····"


def env_key_entries(name):
    found = []
    for base in KEY_ENV_ALIASES.get(name, [name]):
        pat = re.compile(r"^" + re.escape(base) + r"(?:_?(\d+))?$")
        for var, val in os.environ.items():
            m = pat.match(var)
            if m and val.strip():
                num = int(m.group(1)) if m.group(1) else 0
                for part in re.split(r"[,\s]+", val.strip()):
                    if part:
                        found.append((num, var, part))
    found.sort(key=lambda t: (t[0], t[1]))
    return [(k, "env " + v) for _, v, k in found]


def saved_key_list(name):
    d = settings_load()
    out = [k for k in ((d.get("KEYS") or {}).get(name) or []) if isinstance(k, str) and k.strip()]
    legacy = d.get(name)                       # a key saved by an older version of the bot
    if isinstance(legacy, str) and legacy.strip():
        out.append(legacy.strip())
    return out


def key_entries(name):
    """[(key, source)] - website keys first (newest first), then Railway env keys. Duplicates dropped."""
    seen, out = set(), []
    for k in saved_key_list(name):
        if k not in seen:
            seen.add(k)
            out.append((k, "website"))
    for k, src in env_key_entries(name):
        if k not in seen:
            seen.add(k)
            out.append((k, src))
    return out


def key_list(name):
    return [k for k, _ in key_entries(name)]


def first_key(name):
    ks = key_list(name)
    return ks[0] if ks else ""


def refresh_key_globals():
    for name in KEY_NAMES:
        globals()[name] = first_key(name)


def key_ok(name, key):
    with _key_lock:
        _key_state[_kid(name, key)] = {"ok": time.time(), "bad_until": 0, "why": "", "kind": ""}


def key_bad(name, key, e):
    cool = (e.retry + 3) if (e.kind == "rate" and e.retry) else KEY_COOLDOWN.get(e.kind, 300)
    with _key_lock:
        st = _key_state.setdefault(_kid(name, key), {})
        st.update(bad_until=time.time() + cool, why=str(e)[:220], kind=e.kind, bad_at=time.time())
    status(f"🔑 {esc(name)} key {esc(mask_key(key))} failed ({esc(e.kind)}): {esc(str(e)[:110])}"
           + (" - trying the next key." if len(key_list(name)) > 1 else ""))


def key_reset(name=None):
    with _key_lock:
        for n in ([name] if name else KEY_NAMES):
            for k in key_list(n):
                _key_state.pop(_kid(n, k), None)


def with_keys(name, fn):
    """Call fn(key) with each usable key of this provider until one works (KeyFail = try the next)."""
    ks = key_list(name)
    if not ks:
        raise RuntimeError(f"no {name} - add one on the Keys tab")
    now = time.time()
    with _key_lock:
        order = [k for k in ks if _key_state.get(_kid(name, k), {}).get("bad_until", 0) <= now]
    if not order:                               # everything is cooling down: try them anyway, soonest first
        with _key_lock:
            order = sorted(ks, key=lambda k: _key_state.get(_kid(name, k), {}).get("bad_until", 0))
    last = None
    for k in order:
        check_kill()
        try:
            res = fn(k)
            key_ok(name, k)
            return res
        except KeyFail as e:
            key_bad(name, k, e)
            last = e
    raise KeyFail(f"all {len(order)} {name} key(s) failed. Last: {last}", last.kind if last else "invalid",
                  last.retry if last else 0)


def key_check(r, who, limit_codes=(402, 429)):
    """Turn a provider HTTP error into KeyFail (so the next key is tried). Other errors are left to the caller."""
    c = r.status_code
    if c < 400:
        return
    txt = (r.text or "")[:300]
    low = txt.lower()
    msg = f"{who} {c}: {txt[:140]}"
    retry = 0
    try:
        retry = int(float((getattr(r, "headers", {}) or {}).get("retry-after", 0)))
    except Exception:
        pass
    quota = any(w in low for w in ("quota", "credit", "balance", "billing", "insufficient", "per day", "daily"))
    if c in limit_codes or (c in (401, 403) and quota):
        raise KeyFail(msg, "rate" if (c == 429 and not quota) else "credit", retry)
    if c in (401, 403):
        raise KeyFail(msg, "invalid")


def key_rows(name):
    now = time.time()
    rows = []
    for k, src in key_entries(name):
        st = _key_state.get(_kid(name, k), {})
        if st.get("bad_until", 0) > now:
            status_txt, cls = f"{st.get('kind', 'failed')}: {st.get('why', '')[:90]}", "bad"
        elif st.get("ok"):
            status_txt, cls = "worked " + time.strftime("%H:%M", time.localtime(st["ok"])), "ok"
        else:
            status_txt, cls = "not used yet", "idle"
        rows.append({"kid": _kid(name, k), "mask": mask_key(k), "source": src, "status": status_txt, "cls": cls,
                     "removable": src == "website"})
    return rows


def keys_add(name, key):
    key = (key or "").strip()
    if name not in KEY_NAMES or len(key) < 8 or re.search(r"\s", key):
        raise RuntimeError("that doesn't look like a key (no spaces, at least 8 characters)")
    d = settings_load()
    ks = dict(d.get("KEYS") or {})
    lst = [k for k in ks.get(name, []) if k != key]
    lst.insert(0, key)                          # newest first
    ks[name] = lst
    settings_update({"KEYS": ks})
    key_reset(name)


def keys_remove(name, kid):
    d = settings_load()
    ks = dict(d.get("KEYS") or {})
    ks[name] = [k for k in ks.get(name, []) if _kid(name, k) != kid]
    ch = {"KEYS": ks}
    if isinstance(d.get(name), str) and _kid(name, d[name]) == kid:
        ch[name] = None
    settings_update(ch)


def keys_alerts():
    """Providers whose every key is currently failing (shown as a banner with a quick fix)."""
    out = []
    now = time.time()
    for name, label, _ in KEY_PROVIDERS:
        ks = key_list(name)
        if not ks:
            continue
        sts = [_key_state.get(_kid(name, k), {}) for k in ks]
        if all(s.get("bad_until", 0) > now for s in sts):
            out.append((name, label, sts[0].get("why", "")))
    return out


# ---------------------------------------------------------------------------
# Persistence: settings.json + optional GitHub backup (for Railway without a volume)
# ---------------------------------------------------------------------------
_settings_lock = threading.RLock()
_backup_lock = threading.Lock()


def _convert(kind, v):
    if kind == "int":
        return int(float(v))
    if kind == "float":
        return float(v)
    if kind == "bool":
        return str(v).lower() in ("1", "true", "yes", "on")
    return str(v).strip()


def settings_load():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def apply_settings(d=None):
    """Put saved settings over the env defaults (an empty/removed setting falls back to the env value)."""
    d = settings_load() if d is None else d
    for key, _, kind in SETTING_FIELDS:
        val = ENV_DEFAULTS[key]
        if d.get(key) not in (None, ""):
            try:
                val = _convert(kind, d[key])
            except (ValueError, TypeError):
                pass
        globals()[key] = val


def settings_update(changes):
    """changes: {name: value}; None or '' removes the saved value."""
    with _settings_lock:
        d = settings_load()
        for k, v in changes.items():
            if v is None or v == "":
                d.pop(k, None)
            else:
                d[k] = v
        tmp = SETTINGS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, SETTINGS_FILE)
        apply_settings(d)
        refresh_key_globals()
    backup_file(SETTINGS_FILE)


def settings_from_form(f):
    ch = {}
    for key, _, kind in SETTING_FIELDS:
        if kind == "bool":
            ch[key] = "1" if key in f else "0"
            continue
        v = (f.get(key) or [""])[0].strip()
        if kind == "secret":
            if v == "":
                continue                      # blank = keep what is saved
            if v == "-":
                ch[key] = None                # "-" = clear it
                continue
        elif kind in ("int", "float") and v:
            try:
                _convert(kind, v)
            except ValueError:
                continue
        ch[key] = v
    settings_update(ch)


def backup_on():
    return bool(BACKUP_REPO and BACKUP_TOKEN)


def _gh(method, path, accept="application/vnd.github+json", **kw):
    return http(method, f"https://api.github.com/repos/{BACKUP_REPO}/contents/{path}",
                headers={"Authorization": f"Bearer {BACKUP_TOKEN}", "Accept": accept,
                         "X-GitHub-Api-Version": "2022-11-28"}, timeout=60, **kw)


def _backup_ok(name):
    return bool(re.match(r"^(settings\.json|krea_oauth\.json|youtube_cookies\.txt|reference\.(png|jpe?g|webp))$", name))


def backup_file(path):
    """Copy one saved file into the private backup repo (background, skipped if unchanged)."""
    if not backup_on() or not os.path.isfile(path) or not _backup_ok(os.path.basename(path)):
        return

    def work():
        with _backup_lock:
            name = os.path.basename(path)
            try:
                with open(path, "rb") as f:
                    data = f.read()
                blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
                rp = f"{BACKUP_PATH}/{name}"
                r = _gh("GET", f"{rp}?ref={urllib.parse.quote(BACKUP_BRANCH)}")
                sha = r.json().get("sha") if r.status_code == 200 else None
                if sha == blob:
                    return
                body = {"message": f"faceless-bot backup: {name}", "branch": BACKUP_BRANCH,
                        "content": base64.b64encode(data).decode()}
                if sha:
                    body["sha"] = sha
                r = _gh("PUT", rp, json=body)
                if r.status_code >= 300:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
            except BaseException as e:
                status(f"⚠️ Backup of {esc(name)} failed: {esc(str(e)[:200])}")

    threading.Thread(target=work, daemon=True).start()


def backup_delete(name):
    if not backup_on() or not _backup_ok(name):
        return

    def work():
        with _backup_lock:
            try:
                rp = f"{BACKUP_PATH}/{name}"
                r = _gh("GET", f"{rp}?ref={urllib.parse.quote(BACKUP_BRANCH)}")
                if r.status_code != 200:
                    return
                _gh("DELETE", rp, json={"message": f"faceless-bot backup: remove {name}",
                                        "sha": r.json()["sha"], "branch": BACKUP_BRANCH})
            except BaseException as e:
                print(f"backup delete failed: {e}", flush=True)

    threading.Thread(target=work, daemon=True).start()


def restore_from_github():
    """On start: bring back any saved file that is missing locally (fresh deploy without a volume)."""
    if not backup_on():
        return
    try:
        r = _gh("GET", f"{BACKUP_PATH}?ref={urllib.parse.quote(BACKUP_BRANCH)}")
        if r.status_code == 404:
            status("ℹ️ Backup repo reachable, nothing saved there yet (it fills as you save things).")
            return
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:150]}")
        n = 0
        for it in r.json():
            name = it.get("name", "")
            if it.get("type") != "file" or not _backup_ok(name):
                continue
            if name.startswith("reference."):
                if reference_path():
                    continue
            elif os.path.isfile(os.path.join(DATA_DIR, name)):
                continue
            raw = _gh("GET", f"{BACKUP_PATH}/{name}?ref={urllib.parse.quote(BACKUP_BRANCH)}",
                      accept="application/vnd.github.raw+json")
            if raw.status_code == 200:
                with open(os.path.join(DATA_DIR, name), "wb") as f:
                    f.write(raw.content)
                n += 1
        if n:
            status(f"♻️ Restored {n} saved file(s) from the backup repo.")
    except Exception as e:
        status(f"⚠️ Couldn't restore from the backup repo: {esc(str(e)[:200])}")


def storage_status():
    """List of ways your data survives a redeploy (empty = it does NOT)."""
    out = []
    vol = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "")
    if vol and os.path.abspath(DATA_DIR).startswith(os.path.abspath(vol)):
        out.append(f"Railway volume ({vol})")
    if backup_on():
        out.append(f"GitHub backup ({BACKUP_REPO})")
    return out


def volume_on():
    vol = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "")
    return bool(vol and os.path.abspath(DATA_DIR).startswith(os.path.abspath(vol))) or bool(os.environ.get("DATA_DIR"))


def init_storage():
    global PAUSED
    os.makedirs(DATA_DIR, exist_ok=True)
    restore_from_github()
    apply_settings()
    refresh_key_globals()
    if settings_load().get("PAUSED") == "1":
        PAUSED = True
        kill_event.set()
        status("⛔ The bot is PAUSED (kill switch was pressed before). Press Resume on the dashboard to use it again.")
    s = storage_status()
    status("💾 Saved data survives redeploys via: " + ", ".join(s) if s else
           "⚠️ Nothing keeps your saved data across redeploys yet - add a Railway volume or set BACKUP_REPO.")
    if not volume_on():
        status("⚠️ The video bank and image bank only survive a redeploy with a Railway Volume (the GitHub backup is too small for them).")


# ---------------------------------------------------------------------------
# YouTube: latest video (or a direct video link), subtitles, clip
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
    check_kill()
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


VIDEO_LINK_RE = re.compile(r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^#\s]*&)?v=|shorts/|live/|embed/|v/))([\w-]{11})")


def direct_video(text):
    """If the text is a link to ONE video, return it (title via oEmbed). Otherwise None."""
    m = VIDEO_LINK_RE.search((text or "").strip())
    if not m:
        return None
    vid_id = m.group(1)
    url = f"https://www.youtube.com/watch?v={vid_id}"
    title = vid_id
    try:
        r = http("GET", "https://www.youtube.com/oembed", params={"url": url, "format": "json"}, timeout=20, tries=2)
        if r.status_code == 200:
            title = html.unescape(r.json().get("title") or title)
    except Exception as e:
        print(f"oembed failed: {e}", flush=True)
    return {"id": vid_id, "title": title, "url": url}


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
    d = direct_video(handle)              # a direct video link wins
    if d:
        return d
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
# VIDEO BANK: the source video is downloaded once and kept (Vid Kraken first, yt-dlp as the fallback)
# ---------------------------------------------------------------------------
VK_BASE = "https://vidkraken.com/api/v2"
VK_DONE_KEYS = ["downloadUrl", "download_url", "fileUrl", "file_url", "cdnUrl", "cdn_url"]


def bank_find(vid_id):
    """(path, meta) of the banked source video, or ('', {})."""
    try:
        for n in os.listdir(VBANK_DIR):
            if n.startswith(vid_id + ".") and not n.endswith((".json", ".part")):
                p = os.path.join(VBANK_DIR, n)
                if os.path.getsize(p) > 10_000:
                    try:
                        with open(os.path.join(VBANK_DIR, vid_id + ".json"), encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        meta = {}
                    return p, meta
    except OSError:
        pass
    return "", {}


def bank_put(vid, src_path, complete, seconds, how):
    for n in os.listdir(VBANK_DIR):                      # replace any older copy of this video
        if n.startswith(vid["id"] + "."):
            try:
                os.remove(os.path.join(VBANK_DIR, n))
            except OSError:
                pass
    ext = os.path.splitext(src_path)[1] or ".mp4"
    dst = os.path.join(VBANK_DIR, vid["id"] + ext)
    shutil.move(src_path, dst)
    with open(os.path.join(VBANK_DIR, vid["id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump({"title": vid["title"], "url": vid["url"], "complete": complete,
                   "seconds": seconds, "how": how, "saved": int(time.time())}, f)
    return dst


def vidkraken_download(video_url, dest):
    """Vid Kraken: submit, poll, stream the file to dest (360p keeps it small; the frames only need 360p).
    Several keys are rotated (the free plan is about 10 downloads a day per key)."""
    def vk_check(r):
        c = r.status_code
        if c == 401:
            raise KeyFail(f"Vid Kraken 401: {r.text[:120]}", "invalid")
        if c in (402, 403, 429):
            raise KeyFail(f"Vid Kraken limit reached ({c}: {r.text[:100]})", "credit")
        if c >= 300:
            raise RuntimeError(f"Vid Kraken {c}: {r.text[:160]}")

    def go(key):
        hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        r = http("POST", f"{VK_BASE}/download", headers=hdr, json={"url": video_url, "format": "360"}, timeout=60, tries=1)
        vk_check(r)
        j = r.json()
        job = _find_key(j, ["jobId", "job_id", "id"])
        if not job:
            raise RuntimeError(f"Vid Kraken gave no job id: {str(j)[:160]}")
        t0 = time.time()
        url = None
        while time.time() - t0 < 900:
            check_kill()
            p = http("GET", f"{VK_BASE}/download/{job}", headers=hdr, timeout=30, tries=2)
            vk_check(p)
            jj = p.json()
            url = _find_key(jj, VK_DONE_KEYS)
            st = str(jj.get("status") or "").upper()
            if url:
                break
            if st in ("FAILED", "ERROR", "CANCELLED", "CANCELED"):
                raise RuntimeError(f"Vid Kraken job {st}: {str(jj.get('error') or jj.get('errorCode') or jj)[:160]}")
            time.sleep(3)
        if not url:
            raise RuntimeError("Vid Kraken timed out")
        tmp = dest + ".part"
        with requests.get(url, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Vid Kraken file download failed ({resp.status_code})")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(1024 * 1024):
                    check_kill()
                    f.write(chunk)
        os.replace(tmp, dest)

    with_keys("VIDKRAKEN_API_KEY", go)


def get_source_video(vid, seconds, work):
    """Banked copy -> Vid Kraken -> yt-dlp -> nothing. Returns a path or ''."""
    path, meta = bank_find(vid["id"])
    if path and (meta.get("complete") or float(meta.get("seconds") or 0) >= seconds - 1):
        status("🏦 Using the source video from the video bank (no download used).")
        return path
    if key_list("VIDKRAKEN_API_KEY"):
        try:
            status("⬇️ Downloading the source video from Vid Kraken (kept in the bank for next time) ...")
            tmp = os.path.join(work, "vk_src.mp4")
            vidkraken_download(vid["url"], tmp)
            if os.path.getsize(tmp) < 10_000:
                raise RuntimeError("downloaded file is empty")
            return bank_put(vid, tmp, True, 0, "vidkraken")
        except Exception as e:
            print(f"Vid Kraken failed: {e}", flush=True)
            status(f"⚠️ Vid Kraken didn't work ({esc(str(e)[:160])}). Trying the YouTube downloader instead ...")
    try:
        fetch_cookies_from_github()
        src = yt_call(download_clip, vid["url"], work, seconds)
        return bank_put(vid, src, False, seconds, "yt-dlp")
    except Exception as e:
        print(f"yt-dlp video download failed: {e}", flush=True)
        status(f"⚠️ Couldn't download the video ({esc(str(e)[:160])}). Images will be drawn from the text only.")
    return ""


# ---------------------------------------------------------------------------
# Watching the video: scenes, key frames, "is this frame new?"
# ---------------------------------------------------------------------------
def detect_cuts(video, limit):
    r = sp_run(["ffmpeg", "-i", video, "-t", str(limit), "-an",
                "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-f", "null", "-"], timeout=900)
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
    r = sp_run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=9:8:flags=area,format=gray",
                "-f", "rawvideo", "-"], timeout=30, text=False)
    b = r.stdout
    if len(b) < 72:
        return None
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (1 if b[y * 9 + x] > b[y * 9 + x + 1] else 0)
    return bits


def make_frames(src, scenes, unique, framedir):
    """One real frame (middle of the scene) per distinct look. Returns {scene index: jpg path}."""
    os.makedirs(framedir, exist_ok=True)
    total = duration(src)
    out = {}
    for i in unique:
        check_kill()
        fp = os.path.join(framedir, f"f{i:03d}.jpg")
        if not os.path.isfile(fp):
            a, b = scenes[i][0], scenes[i][1]
            t = (a + b) / 2
            if total:
                t = max(0.0, min(t, total - 0.5))
            try:
                grab_frame(src, t, fp)
            except RuntimeError as e:
                print(f"frame {i} failed: {e}", flush=True)
                continue
        if os.path.isfile(fp) and os.path.getsize(fp) > 500:
            out[i] = fp
    return out


# ---------------------------------------------------------------------------
# LLM: Gemini (watches the video), Groq (rewrite / titles / frame descriptions), Anthropic (backup)
# ---------------------------------------------------------------------------
_gemini_state = {"no_thinking": False, "picked": False}


def pick_gemini_model():
    """Newest plain 'gemini-X-flash' model this key can use."""
    r = http("GET", "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
             headers={"x-goog-api-key": first_key("GEMINI_API_KEY")}, timeout=30)
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
    """One Gemini generateContent call. Rotates through your Gemini keys; handles a removed model name."""
    multi = len(key_list("GEMINI_API_KEY")) > 1

    def go(key):
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
                     headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                     json=body, timeout=timeout, tries=1 if multi else 5)
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
        low = r.text.lower()
        if r.status_code in (400, 401) and ("api key" in low or "api_key" in low):
            raise KeyFail(f"Gemini {r.status_code}: {r.text[:140]}", "invalid")
        if r.status_code == 403 and ("api key" in low or "billing" in low or "leaked" in low):
            raise KeyFail(f"Gemini 403: {r.text[:140]}", "invalid")
        if r.status_code == 429:
            key_check(r, "Gemini")
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

    return with_keys("GEMINI_API_KEY", go)


def llm(system, text, max_tokens=4000, image_b64=None):
    """One text (optionally + one JPEG image) in, text out. Gemini if it has a key, otherwise Anthropic."""
    if key_list("GEMINI_API_KEY"):
        parts = [{"text": text}]
        if image_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
        return gemini_call(system, parts, max_tokens)
    content = [{"type": "text", "text": text}]
    if image_b64:
        content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}})

    def go(key):
        r = http("POST", "https://api.anthropic.com/v1/messages",
                 headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                 json={"model": CLAUDE_MODEL, "max_tokens": max_tokens, "system": system,
                       "messages": [{"role": "user", "content": content}]}, timeout=180, tries=2)
        key_check(r, "Anthropic")
        if r.status_code != 200:
            raise RuntimeError(f"Claude API {r.status_code}: {r.text[:200]}")
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()

    return with_keys("ANTHROPIC_API_KEY", go)


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _groq_once(model, messages, max_tokens, temperature):
    def go(key):
        r = http("POST", GROQ_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                 json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
                 timeout=180, tries=1)
        low = r.text.lower()
        if r.status_code in (400, 404, 413) and ("model" in low or "decommission" in low or "too large" in low):
            raise ModelRejected(f"Groq model {model}: {r.text[:140]}")
        key_check(r, "Groq")
        if r.status_code != 200:
            raise RuntimeError(f"Groq {r.status_code}: {r.text[:200]}")
        txt = (((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if not txt:
            raise RuntimeError("Groq returned empty text")
        return txt
    return with_keys("GROQ_API_KEY", go)


def groq_chat(system, user, max_tokens=3000, temperature=0.6, image_b64=None):
    """Groq chat completion: every key x every model in the list (each model has its own rate limit)."""
    if image_b64:
        models = [GROQ_VISION_MODEL.strip()] if GROQ_VISION_MODEL.strip() else []
        content = [{"type": "text", "text": user},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}}]
    else:
        models = [m.strip() for m in GROQ_MODELS.split(",") if m.strip()]
        content = user
    if not models:
        raise RuntimeError("no Groq model set")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    last = None
    for attempt in range(3):
        wait = 0
        for model in models:
            try:
                return _groq_once(model, messages, max_tokens, temperature)
            except ModelRejected as e:
                last = e
            except KeyFail as e:
                last = e
                if e.kind != "rate":
                    raise
                wait = max(wait, e.retry or 10)
        if wait and attempt < 2:
            wait = min(wait, 40)
            status(f"⏳ Groq is rate-limiting, waiting {wait}s ...")
            for _ in range(int(wait)):
                check_kill()
                time.sleep(1)
            continue
        break
    raise last or RuntimeError("Groq failed")


def llm_text(system, text, max_tokens=3000):
    """Text-only AI call: Groq first (cheap), Gemini/Anthropic as the backup if you allow it."""
    if USE_GROQ and key_list("GROQ_API_KEY"):
        try:
            return groq_chat(system, text, max_tokens)
        except Exception as e:
            print(f"Groq text failed: {e}", flush=True)
            if not (TEXT_FALLBACK and (key_list("GEMINI_API_KEY") or key_list("ANTHROPIC_API_KEY"))):
                raise
            status(f"⚠️ Groq failed ({esc(str(e)[:120])}) - using Gemini/Anthropic for this step.")
    return llm(system, text, max_tokens)


def llm_vision(system, text, image_b64, max_tokens=400):
    """One image + text in, text out: Groq vision first, Gemini/Anthropic as the backup."""
    if USE_GROQ and key_list("GROQ_API_KEY") and GROQ_VISION_MODEL.strip():
        try:
            return groq_chat(system, text, max_tokens, image_b64=image_b64)
        except Exception as e:
            print(f"Groq vision failed: {e}", flush=True)
            if not (TEXT_FALLBACK and (key_list("GEMINI_API_KEY") or key_list("ANTHROPIC_API_KEY"))):
                raise
            status(f"⚠️ Groq vision failed ({esc(str(e)[:100])}) - using Gemini/Anthropic for this frame.")
    return llm(system, text, max_tokens, image_b64=image_b64)


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


def _watch_range(video_url, t0, t1):
    """Gemini watches seconds t0..t1 of the video. Returns rows [start, end, narration, visual, same]
    with times counted from the start of the WHOLE video."""
    extra = ""
    if t0 > 0:
        extra = (f"\n\nYou are only shown seconds {int(t0)} to {int(t1)} of the video. Give every start and end "
                 f"as seconds counted from the START OF THE WHOLE VIDEO (the first scene begins at about {int(t0)}).")
    parts = [{"file_data": {"file_uri": video_url},
              "video_metadata": {"start_offset": f"{int(t0)}s", "end_offset": f"{int(t1)}s"}},
             {"text": WATCH_PROMPT.replace("{N}", str(int(t1 - t0))) + extra}]
    raw = gemini_call("", parts, max_tokens=16000, json_mode=True, temperature=0.3, timeout=600)
    rows = []
    for it in parse_json_list(raw):
        try:
            a, b = _secs(it["start"]), _secs(it["end"])
        except (KeyError, ValueError, TypeError):
            continue
        rows.append([a, b, re.sub(r"\s+", " ", str(it.get("narration") or "")).strip(),
                     str(it.get("visual") or "").strip(), bool(it.get("same_as_previous"))])
    if t0 > 0 and rows and min(r[0] for r in rows) < t0 - 1.5:     # Gemini counted from the clip start: shift
        for r in rows:
            r[0] += t0
            r[1] += t0
    out = []
    for r in rows:
        if r[0] >= t1 + 1:
            continue
        r[1] = min(r[1], t1)
        if r[1] > r[0]:
            out.append(r)
    return out


def gemini_watch(video_url, seconds, chunk=180):
    """Let Gemini open the YouTube link itself and describe the first `seconds`
    (long videos are watched in chunks). Returns (scenes[[start,end,text]], visuals[prompt], owner)."""
    seconds = int(seconds)
    flat = []
    t = 0
    while t < seconds:
        check_kill()
        t1 = min(seconds, t + chunk)
        if seconds - t1 < 20:
            t1 = seconds
        if seconds > chunk:
            status(f"👀 Gemini is watching {t}s - {t1}s of {seconds}s ...")
        part, last = None, None
        for _ in range(2):
            try:
                part = _watch_range(video_url, t, t1)
                break
            except Exception as e:
                last = e
                time.sleep(5)
        if part is None:
            raise RuntimeError(f"Gemini failed on {t}s-{t1}s: {str(last)[:200]}")
        flat += part
        t = t1
    flat.sort(key=lambda r: r[0])
    scenes, visuals, same = [], [], []
    for a, b, text, vis, sm in flat:
        if not text and scenes:                        # silent stretch joins the previous scene
            scenes[-1][1] = max(scenes[-1][1], b)
            continue
        scenes.append([a, b, text])                    # (a silent start is merged into the next scene below)
        visuals.append(vis)
        same.append(sm)
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

META_SYSTEM = """You write YouTube upload details for a video whose script is given.
Return ONLY a JSON object: {"title": "...", "description": "...", "tags": ["...", "..."]}
- title: max 90 characters, clear and honest, in your own words (do not copy the source title).
- description: 2 to 4 short plain sentences saying what the video covers, then a blank line and 3 relevant hashtags.
- tags: 8 to 12 short keywords.
Only claim what the script says. No emojis."""


def parse_json_list(text):
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b < 0:
        raise ValueError("no JSON list in reply")
    return json.loads(text[a:b + 1])


def parse_json_obj(text):
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("no JSON object in reply")
    return json.loads(text[a:b + 1])


def rewrite_scenes(scenes, title):
    new = {}
    for start in range(0, len(scenes), 15):
        check_kill()
        chunk = [{"id": i, "text": scenes[i][2]} for i in range(start, min(start + 15, len(scenes)))]
        msg = f"Video title (context only): {title}\n\n{json.dumps(chunk, ensure_ascii=False)}"
        for attempt in range(2):
            try:
                for d in parse_json_list(llm_text(REWRITE_SYSTEM, msg, 6000)):
                    new[int(d["id"])] = str(d["text"]).strip()
                break
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                if attempt:
                    raise RuntimeError(f"couldn't parse the AI's rewrite: {e}")
    return [new.get(i) or scenes[i][2] for i in range(len(scenes))]


def make_metadata(new_text, src_title):
    """Title / description / tags for the upload (falls back to plain values if the AI fails)."""
    script = " ".join(new_text)[:6000]
    try:
        d = parse_json_obj(llm_text(META_SYSTEM, f"Source video title (context only): {src_title}\n\nScript:\n{script}", 1500))
        title = re.sub(r"\s+", " ", str(d.get("title") or "")).strip()[:100]
        desc = str(d.get("description") or "").strip()[:4900]
        tags = [re.sub(r"\s+", " ", str(t)).strip().lstrip("#")[:100] for t in (d.get("tags") or []) if str(t).strip()]
        if title and desc:
            return {"title": title, "description": desc, "tags": tags[:15]}
    except Exception as e:
        print(f"metadata AI failed: {e}", flush=True)
    return {"title": (src_title or "Video")[:100], "description": script[:400], "tags": []}


DESCRIBE_TEXT_SYSTEM = """You write image-generation prompts.
You get the narration of one scene from an explainer video. Write ONE prompt (max 70 words) for an animated
illustration that shows what the narration is about: main subject(s), pose and position, setting, key objects,
mood and colors. No text or lettering in the image. Do NOT name the art style (a style reference image
supplies it). Do not name real people; use generic characters. Output ONLY the prompt."""


def describe_frame(frame_path, narration):
    if not frame_path:     # couldn't watch the video: draw from the narration alone
        return llm_text(DESCRIBE_TEXT_SYSTEM, f"Narration: {narration[:500]}", 300)
    with open(frame_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return llm_vision(DESCRIBE_SYSTEM, f"Narration for this scene: {narration[:400]}", b64, 300)


# ---------------------------------------------------------------------------
# Krea: style image + (optional) real frame -> image (API key route)
# ---------------------------------------------------------------------------
CREDIT_WORDS = ("compute", "credit", "insufficient", "billing", "balance", "quota", "402")
FRAME_KEYS = ["image_url", "image_urls", "init_image_url"]     # names tried (in order) for the frame reference


def ladder_models():
    """Krea models to try, cheapest first. The medium model (KREA_MODEL) is always last."""
    if not USE_CHEAP_KREA:
        return [KREA_MODEL]
    out = []
    for m in [x.strip() for x in str(KREA_LADDER).split(",") if x.strip()]:
        if m != KREA_MODEL and m not in out:
            out.append(m)
    return out + [KREA_MODEL]


def krea_headers():
    return {"Authorization": f"Bearer {getattr(_tl, 'krea_key', None) or first_key('KREA_API_KEY')}"}


def _data_uri(path):
    with open(path, "rb") as f:
        raw = f.read()
    return f"data:{mimetypes.guess_type(path)[0] or 'image/jpeg'};base64," + base64.b64encode(raw).decode()


_asset_url = {}


def krea_asset_url(p):
    """Upload an image to Krea once (per key), return its URL."""
    if not p:
        return ""
    key = (p, os.path.getmtime(p), getattr(_tl, 'krea_key', None))
    if key not in _asset_url:
        with open(p, "rb") as f:
            r = http("POST", f"{KREA_BASE}/assets", headers=krea_headers(),
                     files={"file": (os.path.basename(p), f, mimetypes.guess_type(p)[0] or "image/png")},
                     data={"description": "reference"})
        if r.status_code in (401, 402, 429) or (r.status_code == 403 and "model" not in r.text.lower()):
            key_check(r, "Krea")
        if r.status_code >= 300:
            raise RuntimeError(f"Krea asset upload failed: {r.status_code} {r.text[:200]}")
        _asset_url[key] = r.json()["image_url"]
    return _asset_url[key]


def krea_reference_url():
    return krea_asset_url(reference_path())


def krea_wait(job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        check_kill()
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


# Different Krea models name the size option differently; try them in order and remember the first accepted.
SIZE_VARIANTS = [{"aspect_ratio": "16:9", "resolution": "1K"}, {"aspect_ratio": "16:9"},
                 {"width": 1344, "height": 768}, {}]
_kstate = {}      # model -> {"size": variant index, "frame": key | False | None(unknown)}


def krea_image(model, prompt, ref_url, frame_url, dest):
    """One image from one model. Needs the style reference accepted; the frame is used when the model takes it.
    Raises ModelRejected when the model refuses the request (so the ladder moves on)."""
    st = _kstate.setdefault(model, {"size": 0, "frame": None})
    err = ""
    for vi in range(st["size"], len(SIZE_VARIANTS)):
        base = {"prompt": prompt, **SIZE_VARIANTS[vi]}
        if ref_url:
            base["image_style_references"] = [{"url": ref_url, "strength": STYLE_STRENGTH}]
        tries = []
        if frame_url and st["frame"] is not False:
            for k in ([st["frame"]] if st["frame"] else FRAME_KEYS):
                tries.append((k, {**base, k: ([frame_url] if k == "image_urls" else frame_url)}))
        tries.append((False, base))
        for k, body in tries:
            r = http("POST", f"{KREA_BASE}/generate/image/{model}", headers=krea_headers(), json=body, timeout=60)
            if r.status_code in (401, 402, 429) or (r.status_code == 403 and not any(w in r.text.lower() for w in ("model", "plan", "upgrade", "access", "not available"))):
                key_check(r, "Krea")
            if r.status_code in (400, 403, 404, 422):
                err = f"{r.status_code} {r.text[:200]}"
                continue
            if r.status_code >= 300:
                raise RuntimeError(f"Krea {r.status_code}: {r.text[:200]}")
            st["size"] = vi
            if frame_url and st["frame"] is None:
                st["frame"] = k
                status(f"ℹ️ {esc(model)}: " + ("takes the frame as a 2nd reference image." if k else
                                              "doesn't take a frame reference, using the style image only."))
            url = krea_wait(r.json()["job_id"])
            img = http("GET", url, timeout=120)
            if img.status_code != 200:
                raise RuntimeError(f"couldn't download Krea image ({img.status_code})")
            with open(dest, "wb") as f:
                f.write(img.content)
            return
    raise ModelRejected(f"Krea rejected {model}: {err}")


def krea_image_keyed(model, prompt, frame_path, dest):
    """krea_image with key rotation (a Krea job is polled with the same key that created it)."""
    def go(key):
        _tl.krea_key = key
        try:
            krea_image(model, prompt, krea_reference_url(), krea_asset_url(frame_path) if frame_path else "", dest)
        finally:
            _tl.krea_key = None
    with_keys("KREA_API_KEY", go)


# ---------------------------------------------------------------------------
# Voice-over: Fish Audio -> ElevenLabs -> (Gemini, optional) -> free Microsoft voice
# ---------------------------------------------------------------------------
def fish_tts(text, dest):
    def go(key):
        body = {"text": text, "format": "mp3"}
        if FISH_VOICE_ID:
            body["reference_id"] = FISH_VOICE_ID
        r = http("POST", "https://api.fish.audio/v1/tts",
                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                          "model": FISH_MODEL}, json=body, timeout=180, tries=2)
        key_check(r, "Fish Audio")
        if r.status_code != 200:
            raise RuntimeError(f"Fish Audio {r.status_code}: {r.text[:200]}")
        with open(dest, "wb") as f:
            f.write(r.content)
    with_keys("FISH_API_KEY", go)


def eleven_tts(text, dest):
    """ElevenLabs voice-over (mp3)."""
    def go(key):
        r = http("POST", f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}?output_format=mp3_44100_128",
                 headers={"xi-api-key": key, "Content-Type": "application/json"},
                 json={"text": text, "model_id": ELEVEN_MODEL}, timeout=180, tries=2)
        key_check(r, "ElevenLabs")
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:200]}")
        with open(dest, "wb") as f:
            f.write(r.content)
    with_keys("ELEVEN_API_KEY", go)


def edge_tts_save(text, dest):
    """Free Microsoft voice (edge-tts). No key, no credit."""
    import asyncio
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts isn't installed (pip install edge-tts)")
    last = None
    for i in range(3):                    # the unofficial service sometimes hiccups
        check_kill()
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
             headers={"x-goog-api-key": first_key("GEMINI_API_KEY")}, timeout=30)
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
                 headers={"x-goog-api-key": first_key("GEMINI_API_KEY"), "Content-Type": "application/json"},
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


def voice_sig():
    return "|".join([str(bool(key_list("FISH_API_KEY"))), FISH_VOICE_ID, str(bool(key_list("ELEVEN_API_KEY"))), ELEVEN_VOICE_ID,
                     ELEVEN_MODEL, EDGE_VOICE, str(USE_GEMINI_VOICE)])


def make_voiceovers(texts, audios):
    """Returns (voice name, used_backup). Try the voices in order. The first that voices EVERY line wins.
    If a voice fails on the first line or runs out part-way, all lines are redone with the next voice
    (so the voice never changes halfway through the video)."""
    options = []
    if key_list("FISH_API_KEY"):
        options.append(("Fish Audio", fish_tts, WORKERS))
    if key_list("ELEVEN_API_KEY"):
        options.append(("ElevenLabs", eleven_tts, 2))
    if key_list("GEMINI_API_KEY") and USE_GEMINI_VOICE:
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
        return name, k > 0           # k > 0 = a backup voice had to be used


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
                "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-profile:v", "high", "-preset", "veryfast", "-crf", "22",
                "-threads", FFMPEG_THREADS, "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
                "-t", f"{d:.3f}", "-movflags", "+faststart", dst], f"scene {idx + 1}")


def concat(parts, out_path, workdir):
    lst = os.path.join(workdir, "list.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write("file '" + p.replace("'", "'\\''") + "'\n")
    run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                "-movflags", "+faststart", out_path], "join scenes")


def verify_video(path):
    """(ok, note): H.264 video + AAC audio, yuv420p, sane duration - i.e. something every phone and browser plays."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,pix_fmt",
                            "-show_entries", "format=duration,size", "-of", "json", path],
                           capture_output=True, text=True, timeout=60)
        j = json.loads(r.stdout or "{}")
    except Exception as e:
        return False, f"ffprobe failed: {e}"
    st = j.get("streams") or []
    v = next((s for s in st if s.get("codec_type") == "video"), None)
    a = next((s for s in st if s.get("codec_type") == "audio"), None)
    dur = float((j.get("format") or {}).get("duration") or 0)
    size = int((j.get("format") or {}).get("size") or 0)
    if not v or v.get("codec_name") != "h264" or v.get("pix_fmt") != "yuv420p":
        return False, f"video stream is {v.get('codec_name') if v else 'missing'}/{v.get('pix_fmt') if v else '-'}"
    if not a or a.get("codec_name") != "aac":
        return False, f"audio stream is {a.get('codec_name') if a else 'missing'}"
    if dur < 1 or size < 10_000:
        return False, f"too short or tiny ({dur:.1f}s, {size} bytes)"
    return True, f"H.264 + AAC, {int(dur // 60)}:{int(dur % 60):02d}, {size / 1_000_000:.1f} MB"


def concat_reencode(parts, out_path, workdir):
    """Safety net if the stream-copy join isn't clean: re-encode everything into one standard MP4."""
    lst = os.path.join(workdir, "list.txt")
    run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-profile:v", "high",
                "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-threads", FFMPEG_THREADS,
                "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart", out_path],
               "re-encode join")


def check_setup():
    missing = [] if (key_list("KREA_API_KEY") or CF_IMAGE_URL or krea_connected()) else ["a Krea key, CF_IMAGE_URL or a Krea connection"]
    if not (key_list("GEMINI_API_KEY") or key_list("ANTHROPIC_API_KEY")):
        missing.insert(0, "a Gemini key (free at aistudio.google.com) - it watches the video")
    if missing:
        raise RuntimeError("missing settings: " + ", ".join(missing))
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is not installed")
    if not reference_path():
        status("ℹ️ No style reference image uploaded - images will use Krea's default look.")
    if USE_GROQ and not key_list("GROQ_API_KEY"):
        status("ℹ️ No Groq key - text steps use Gemini/Anthropic. Add a Groq key on the Keys tab to save Gemini credits.")
    if AUTO_POST and not key_list("ZERNIO_API_KEY"):
        status("⚠️ Auto-post is ON but there is no Zernio API key - the video will be built but not posted.")
    if FRAME_REFS and not (key_list("VIDKRAKEN_API_KEY") or cookies_path()):
        status("ℹ️ Frame references need a Vid Kraken key (or YouTube cookies) to download the video - otherwise text-only.")


# ---------------------------------------------------------------------------
# Krea through its MCP server + OAuth: spends your normal Krea web credits (not the separate API balance)
# ---------------------------------------------------------------------------
KREA_MCP_URL = "https://api.krea.ai/mcp"
KREA_OAUTH_FILE = os.path.join(DATA_DIR, "krea_oauth.json")
_oauth_lock = threading.RLock()
_oauth_pending = {}
_mcp = {"sid": None, "tools": None, "lock": threading.RLock(), "n": 0}
_mcp_size = {}        # model -> size variant index
_mcp_frame = {}       # model -> {"on": key | False | None}
_mcp_resolved = {}    # wanted model -> the exact id Krea accepted


def oauth_load():
    try:
        with open(KREA_OAUTH_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    if KREA_OAUTH_BLOB.strip():
        try:
            return json.loads(base64.b64decode(KREA_OAUTH_BLOB.strip()).decode())
        except Exception:
            return None
    return None


def oauth_save(d):
    with open(KREA_OAUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f)
    backup_file(KREA_OAUTH_FILE)


def krea_connected():
    d = oauth_load()
    return bool(d and (d.get("refresh_token") or d.get("access_token")))


def krea_blob():
    d = oauth_load()
    return base64.b64encode(json.dumps(d).encode()).decode() if d else ""


def _json_get(url):
    try:
        r = requests.get(url, timeout=20, headers={"Accept": "application/json"})
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def oauth_discover():
    pu = urllib.parse.urlparse(KREA_MCP_URL)
    origin = f"{pu.scheme}://{pu.netloc}"
    meta = None
    try:                                   # the 401 answer tells us where the metadata lives
        r = requests.post(KREA_MCP_URL, timeout=20,
                          headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                          json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "faceless-bot", "version": "1"}}})
        m = re.search(r'resource_metadata="([^"]+)"', r.headers.get("WWW-Authenticate", ""))
        if m:
            meta = _json_get(m.group(1))
    except Exception:
        pass
    if not meta:
        for u in (origin + "/.well-known/oauth-protected-resource" + pu.path,
                  origin + "/.well-known/oauth-protected-resource"):
            meta = _json_get(u)
            if meta:
                break
    issuer = ((meta or {}).get("authorization_servers") or [origin])[0].rstrip("/")
    ip = urllib.parse.urlparse(issuer)
    asm = None
    for u in (issuer + "/.well-known/oauth-authorization-server",
              f"{ip.scheme}://{ip.netloc}/.well-known/oauth-authorization-server{ip.path}",
              issuer + "/.well-known/openid-configuration"):
        asm = _json_get(u)
        if asm and asm.get("authorization_endpoint") and asm.get("token_endpoint"):
            break
        asm = None
    if not asm:
        raise RuntimeError("couldn't find Krea's sign-in endpoints (OAuth discovery failed)")
    return asm, (meta or {})


def oauth_begin(base):
    """Start the one-time Krea sign-in. Returns the Krea URL to send the browser to."""
    asm, meta = oauth_discover()
    redirect = base + "/krea/callback"
    reg = asm.get("registration_endpoint")
    if not reg:
        raise RuntimeError("Krea doesn't offer automatic app registration for sign-in")
    r = requests.post(reg, timeout=30, json={
        "client_name": "Faceless Bot", "redirect_uris": [redirect],
        "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
        "token_endpoint_auth_method": "none"})
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Krea app registration failed {r.status_code}: {r.text[:200]}")
    cl = r.json()
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    _oauth_pending[state] = {"verifier": verifier, "redirect": redirect, "client_id": cl["client_id"],
                             "client_secret": cl.get("client_secret", ""), "token_endpoint": asm["token_endpoint"]}
    q = {"response_type": "code", "client_id": cl["client_id"], "redirect_uri": redirect,
         "code_challenge": challenge, "code_challenge_method": "S256", "state": state, "resource": KREA_MCP_URL}
    scopes = meta.get("scopes_supported") or asm.get("scopes_supported") or []
    if scopes:
        q["scope"] = " ".join(scopes)
    sep = "&" if "?" in asm["authorization_endpoint"] else "?"
    return asm["authorization_endpoint"] + sep + urllib.parse.urlencode(q)


def _store_tokens(d, tok):
    d["access_token"] = tok["access_token"]
    if tok.get("refresh_token"):
        d["refresh_token"] = tok["refresh_token"]
    d["expires_at"] = time.time() + int(tok.get("expires_in") or 3600)
    oauth_save(d)


def oauth_finish(code, state):
    pend = _oauth_pending.pop(state, None)
    if not pend:
        raise RuntimeError("this sign-in attempt expired - tap Connect Krea again")
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": pend["redirect"],
            "client_id": pend["client_id"], "code_verifier": pend["verifier"], "resource": KREA_MCP_URL}
    if pend["client_secret"]:
        data["client_secret"] = pend["client_secret"]
    r = requests.post(pend["token_endpoint"], data=data, headers={"Accept": "application/json"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Krea token exchange failed {r.status_code}: {r.text[:200]}")
    d = {"client_id": pend["client_id"], "client_secret": pend["client_secret"],
         "token_endpoint": pend["token_endpoint"]}
    _store_tokens(d, r.json())


def oauth_token(force=False):
    with _oauth_lock:
        d = oauth_load()
        if not d:
            raise RuntimeError("Krea isn't connected - tap 'Connect Krea' on the dashboard")
        if not force and d.get("access_token") and d.get("expires_at", 0) - 60 > time.time():
            return d["access_token"]
        if not d.get("refresh_token"):
            raise RuntimeError("Krea sign-in expired - tap 'Connect Krea' again")
        data = {"grant_type": "refresh_token", "refresh_token": d["refresh_token"],
                "client_id": d["client_id"], "resource": KREA_MCP_URL}
        if d.get("client_secret"):
            data["client_secret"] = d["client_secret"]
        r = requests.post(d["token_endpoint"], data=data, headers={"Accept": "application/json"}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Krea sign-in expired ({r.status_code}) - tap 'Connect Krea' again")
        _store_tokens(d, r.json())
        return d["access_token"]


def _sse_pick(text, rid):
    out = None
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                j = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(j, dict):
                if j.get("id") == rid:
                    return j
                if out is None and ("result" in j or "error" in j):
                    out = j
    return out


def mcp_rpc(method, params=None, notify=False, retry=True):
    with _mcp["lock"]:
        _mcp["n"] += 1
        rid = _mcp["n"]
    body = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        body["params"] = params
    if not notify:
        body["id"] = rid
    h = {"Authorization": f"Bearer {oauth_token()}", "Content-Type": "application/json",
         "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
    if _mcp["sid"] and method != "initialize":
        h["Mcp-Session-Id"] = _mcp["sid"]
    r = http("POST", KREA_MCP_URL, headers=h, json=body, timeout=300, tries=3)
    if r.status_code == 401 and retry:
        oauth_token(force=True)
        return mcp_rpc(method, params, notify, retry=False)
    if r.status_code in (400, 404) and _mcp["sid"] and method != "initialize" and retry:
        with _mcp["lock"]:                 # session expired: start a new one
            _mcp["tools"] = None
            _mcp["sid"] = None
        mcp_ready()
        return mcp_rpc(method, params, notify, retry=False)
    if method == "initialize" and r.headers.get("Mcp-Session-Id"):
        _mcp["sid"] = r.headers["Mcp-Session-Id"]
    if notify:
        return None
    if r.status_code >= 300:
        raise RuntimeError(f"Krea MCP {r.status_code}: {r.text[:250]}")
    j = _sse_pick(r.text, rid) if "text/event-stream" in r.headers.get("Content-Type", "") else r.json()
    if not isinstance(j, dict):
        raise RuntimeError(f"Krea MCP gave an unreadable reply: {r.text[:200]}")
    if j.get("error"):
        raise RuntimeError(f"Krea MCP error: {str(j['error'])[:300]}")
    return j.get("result")


def mcp_ready():
    with _mcp["lock"]:
        if _mcp["tools"] is not None:
            return
        _mcp["sid"] = None
        mcp_rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "faceless-bot", "version": "1"}})
        mcp_rpc("notifications/initialized", {}, notify=True)
        tl = mcp_rpc("tools/list", {}) or {}
        _mcp["tools"] = {t["name"]: t for t in tl.get("tools", [])}


def mcp_tool(name, args):
    mcp_ready()
    res = mcp_rpc("tools/call", {"name": name, "arguments": args}) or {}
    text = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
    if res.get("isError"):
        raise RuntimeError(f"Krea tool error: {text[:300]}")
    data = res.get("structuredContent")
    if data is None:
        try:
            data = json.loads(text)
        except ValueError:
            data = {"text": text}
    return data


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def _find_key(o, keys):
    for d in _walk(o):
        for k in keys:
            if d.get(k) not in (None, ""):
                return d[k]
    return None


def _find_urls(o):
    urls = []
    for d in _walk(o):
        for k in ("urls", "url", "image_url", "output_url"):
            v = d.get(k)
            if isinstance(v, str) and v.startswith("http"):
                urls.append(v)
            elif isinstance(v, list):
                urls += [x for x in v if isinstance(x, str) and x.startswith("http")]
    return urls


def _pick_prop(tool, names):
    props = ((tool or {}).get("inputSchema") or {}).get("properties") or {}
    for n in names:
        if n in props:
            return n
    return None


def _find_tool(tools, patterns, exclude=("list_", "read_", "write_", "delete_", "tag")):
    """First MCP tool whose name contains one of the patterns (file/tag tools are skipped)."""
    for pat in patterns:
        for name in tools:
            n = name.lower()
            if pat in n and not n.startswith(exclude):
                return name
    return None


_ref_uri = {}


def reference_data_uri():
    """Your style image, shrunk, as a data: URI (Krea accepts that as a style reference)."""
    pth = reference_path()
    if not pth:
        return ""
    key = (pth, os.path.getmtime(pth))
    if key not in _ref_uri:
        fd, tmp = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            run_ffmpeg(["ffmpeg", "-y", "-i", pth, "-vf", "scale='min(1024,iw)':-2", "-q:v", "4", tmp], "shrink reference")
            with open(tmp, "rb") as f:
                _ref_uri[key] = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return _ref_uri[key]


def _schema_enum(schema, key):
    for d in _walk(schema or {}):
        pr = d.get(key)
        if isinstance(pr, dict) and isinstance(pr.get("enum"), list):
            return [str(x) for x in pr["enum"]]
    return []


_krea_models_cache = {"ids": None}


def krea_list_models(tools):
    """Ask Krea itself which image models this login can use (its list_models tool)."""
    if _krea_models_cache["ids"] is not None:
        return _krea_models_cache["ids"]
    ids = []
    lm = _find_tool(tools, ["list_models"], exclude=())
    if lm:
        props = (tools[lm].get("inputSchema") or {}).get("properties") or {}
        args = {}
        for k in ("category", "type", "kind", "media_type", "modality"):
            pv = props.get(k)
            if isinstance(pv, dict):
                en = pv.get("enum")
                args[k] = "image" if (not en or "image" in en) else en[0]
        for a_ in ([args, {}] if args else [{}]):
            try:
                data = mcp_tool(lm, a_)
            except RuntimeError as e:
                print(f"KREA list_models failed: {e}", flush=True)
                continue
            txt = json.dumps(data)
            print(f"KREA list_models raw: {txt[:3000]}", flush=True)
            ids = list(dict.fromkeys(re.findall(r"image/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+", txt)))
            if not ids:
                ids = list(dict.fromkeys(re.findall(r"[A-Za-z0-9_.\-]+/krea-[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*", txt)))
            if ids:
                break
    _krea_models_cache["ids"] = ids
    return ids


def _krea_model_candidates(gen, model_key, tools, want_model, strict):
    """Model ids to try for want_model. strict = a ladder model: only that model, never a substitute."""
    want = want_model.replace("image/", "")
    enum = _schema_enum(gen.get("inputSchema"), model_key)
    pool = enum or krea_list_models(tools)

    def w(m):
        return m.replace("image/", "")

    if pool:
        if strict:
            cands = [m for m in pool if w(m) == want]
        else:
            cands = ([m for m in pool if w(m) == want]
                     + [m for m in pool if "krea-2" in m and "medium" in m]
                     + [m for m in pool if "krea-2" in m]
                     + [m for m in pool if "krea" in m.lower()] + pool[:5])
    else:
        parts = want.split("/")
        cands = [want, "image/" + want] if strict else [want, "image/" + want, parts[1] if len(parts) > 1 else want, "krea-2", None]
    out = []
    for m in cands:
        if m not in out:
            out.append(m)
    return out, pool


def _krea_collect(data, tools, jobtool, dest):
    jid = _find_key(data, ["job_id", "jobId", "id"])
    t0 = time.time()
    while True:
        check_kill()
        st = str(_find_key(data, ["status"]) or "").lower()
        if st in ("failed", "cancelled", "canceled", "error"):
            raise RuntimeError(f"Krea job {st}: {str(data)[:200]}")
        urls = _find_urls(_find_key(data, ["result", "output", "outputs"]) or data)
        if urls:
            break
        if not jid or not jobtool:
            raise RuntimeError(f"no image URL in Krea's reply: {str(data)[:200]}")
        if time.time() - t0 > 300:
            raise RuntimeError("Krea job timed out")
        time.sleep(3)
        jk = _pick_prop(tools.get(jobtool), ["jobId", "job_id", "id"]) or "jobId"
        data = mcp_tool(jobtool, {jk: jid})
    img = http("GET", urls[0], timeout=120)
    if img.status_code != 200:
        raise RuntimeError(f"couldn't download Krea image ({img.status_code})")
    with open(dest, "wb") as f:
        f.write(img.content)


def krea_mcp_image(prompt, ref_uri, dest, model_pref=None, frame_uri=None):
    """One image through your Krea web credits. model_pref = a ladder model (strict); None = the medium model."""
    mcp_ready()
    tools = _mcp["tools"]
    gname = _find_tool(tools, ["generate_image", "image_generate", "generate", "create_image", "text_to_image", "image"])
    if not gname:
        full = "; ".join(f"{n}: {json.dumps((t.get('inputSchema') or {}).get('properties') or {})[:300]}"
                         for n, t in tools.items())
        print(f"KREA MCP TOOLS (full): {full}", flush=True)
        raise RuntimeError("Krea MCP has no image tool. Tools: " + ", ".join(tools))
    gen = tools[gname]
    props = (gen.get("inputSchema") or {}).get("properties") or {}
    model_key = _pick_prop(gen, ["model", "model_id", "modelId", "model_path"]) or "model"
    box = _pick_prop(gen, ["input", "inputs", "params", "parameters", "arguments", "body", "payload", "request", "options"])
    if not box and "prompt" not in props:
        for k_, pv_ in props.items():
            if isinstance(pv_, dict) and pv_.get("type") == "object" and k_ != model_key:
                box = k_
                break
    base = {"prompt": prompt}
    if ref_uri:
        base["image_style_references"] = [{"url": ref_uri, "strength": STYLE_STRENGTH}]
    jobtool = _find_tool(tools, ["get_job", "job_status", "check_job", "job"])
    want = model_pref or KREA_MODEL
    strict = want != KREA_MODEL
    if want in _mcp_resolved:
        cands, enum = [_mcp_resolved[want]], []
    else:
        cands, enum = _krea_model_candidates(gen, model_key, tools, want, strict)
    fail = ModelRejected if strict else RuntimeError
    if not cands:
        raise fail(f"Krea doesn't offer {want} on this login")
    mst = _mcp_frame.setdefault(want, {"on": None})
    err = ""
    for model in cands:
        model_bad = False
        for vi in range(_mcp_size.get(want, 0), len(SIZE_VARIANTS)):
            inputs0 = {**base, **SIZE_VARIANTS[vi]}
            tries = []
            if frame_uri and mst["on"] is not False:
                for k in ([mst["on"]] if mst["on"] else FRAME_KEYS):
                    tries.append((k, {**inputs0, k: ([frame_uri] if k == "image_urls" else frame_uri)}))
            tries.append((False, inputs0))
            for k, inputs in tries:
                args = {}
                if model:
                    if box and model_key not in props:
                        inputs = {**inputs, model_key: model}
                    else:
                        args[model_key] = model
                if box:
                    args[box] = inputs
                else:
                    args.update(inputs)
                try:
                    data = mcp_tool(gname, args)
                except RuntimeError as e:
                    msg = str(e)
                    low = msg.lower()
                    if any(w in low for w in CREDIT_WORDS):
                        raise OutOfCredit(msg)
                    err = msg
                    if "model" in low and any(w in low for w in ("unsupported", "invalid", "unknown", "not found", "not supported")):
                        model_bad = True
                        break
                    continue
                _mcp_size[want] = vi
                _mcp_resolved[want] = model
                if frame_uri and mst["on"] is None:
                    mst["on"] = k
                    status(f"ℹ️ {esc(want)}: " + ("takes the frame as a 2nd reference image." if k else
                                                 "doesn't take a frame reference, using the style image only."))
                _krea_collect(data, tools, jobtool, dest)
                return
            if model_bad:
                break
        if not model_bad:
            break
    hint = f" Models Krea lists for you: {', '.join(enum[:30])}." if enum else ""
    raise fail(f"Krea MCP rejected {want}: {err[:300]}{hint}")


def krea_test():
    try:
        status("🧪 Testing the Krea model ladder (each model gets one test image) ...")
        use_mcp = krea_connected()
        if use_mcp:
            with _mcp["lock"]:
                _mcp["tools"] = None
            mcp_ready()
            status("🧪 Connected. Krea tools: " + esc(", ".join(_mcp["tools"]) or "none"))
            ids_ = krea_list_models(_mcp["tools"])
            status("🧪 Models Krea lists: " + (esc(", ".join(ids_[:40])) if ids_ else "none found (raw reply is in the Railway logs)"))
        elif not key_list("KREA_API_KEY"):
            raise RuntimeError("connect Krea or add a Krea API key first")
        fr = reference_path()
        for m in ladder_models():
            check_kill()
            dest = os.path.join(OUT_DIR, "krea_test_" + re.sub(r"\W", "_", m) + ".png")
            try:
                if use_mcp:
                    krea_mcp_image("a friendly cartoon caveman sitting by a campfire at night", reference_data_uri(),
                                   dest, None if m == KREA_MODEL else m, _data_uri(fr) if fr else None)
                    fk = _mcp_frame.get(m, {}).get("on")
                else:
                    krea_image_keyed(m, "a friendly cartoon caveman sitting by a campfire at night", fr or None, dest)
                    fk = _kstate.get(m, {}).get("frame")
                status(f'✅ {esc(m)} works with your style image; frame reference: {"yes" if fk else "no"}. '
                       f'<a href="/out/{os.path.basename(dest)}" target="_blank">test image</a>')
            except OutOfCredit as e:
                status(f"❌ {esc(m)}: out of credits ({esc(str(e)[:150])}). Stopping the test.")
                return
            except ModelRejected as e:
                status(f"⛔ {esc(m)} can't be used (it would be skipped): {esc(str(e)[:200])}")
            except RuntimeError as e:
                status(f"❌ {esc(m)}: {esc(str(e)[:200])}")
    except Killed:
        status("⛔ Test stopped by the kill switch.")
    except Exception as e:
        status(f"❌ Krea test failed: {esc(str(e)[:600])}")


def krea_card():
    if krea_connected():
        extra = ""
        if not storage_status():
            extra = ('<details style="margin-top:12px"><summary>Keep it after redeploys (no storage set up)</summary><div class="sb">'
                     'Best fix: add a Railway volume or set BACKUP_REPO. Otherwise paste this into the '
                     '<code>KREA_OAUTH_BLOB</code> env var. It is a secret login, keep it private.'
                     f'<textarea readonly rows="4" style="margin-top:8px">{esc(krea_blob())}</textarea>'
                     '</div></details>')
        return ('<div class="card"><h2>Krea (your web credits)</h2><div class="hint" style="margin:0 0 8px">'
                '✅ Connected. Images use your normal Krea compute credits first. '
                'Model order: ' + esc(" → ".join(ladder_models())) + '</div>'
                '<div class="acts"><form method="POST" action="/krea/test" style="display:inline"><button type="submit">Test the model ladder</button></form> '
                '<form method="POST" action="/krea/disconnect" style="display:inline"><button class="ghost" type="submit">Disconnect</button></form></div>'
                + extra + '</div>')
    return ('<div class="card"><h2>Krea (your web credits)</h2><div class="hint" style="margin:0 0 8px">'
            'Not connected. Sign in once so the bot can spend your normal Krea credits. On the Krea screen, '
            'pick the workspace that has the credits.</div>'
            '<a class="btn" href="/krea/connect">Connect Krea</a></div>')


def cf_image(prompt, dest):
    """Free images from your Cloudflare Worker (Stable Diffusion XL)."""
    r = http("POST", CF_IMAGE_URL, tries=4, timeout=120,
             headers={"Authorization": f"Bearer {CF_IMAGE_KEY}", "Content-Type": "application/json",
                      "User-Agent": "Mozilla/5.0"},
             json={"prompt": prompt, "width": 1024, "height": 576})
    if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("image"):
        raise RuntimeError(f"Cloudflare image {r.status_code}: {r.text[:150]}")
    with open(dest, "wb") as f:
        f.write(r.content)


def make_images(unique, prompts, imgdir, frames=None):
    """Draw every image with ONE provider/model. Order: Krea credits (cheapest model first, up to medium), then
    the Krea API (same ladder), then the free Cloudflare worker. A model that rejects reference images (or runs
    out part-way) is skipped and everything is redone with the next one, so the style stays consistent."""
    frames = frames or {}
    os.makedirs(imgdir, exist_ok=True)

    def hint(pr):
        return pr + (". " + STYLE_HINT if STYLE_HINT else "")

    def mk_mcp(m):
        mp = None if m == KREA_MODEL else m

        def fn(i, d):
            fr = frames.get(i)
            krea_mcp_image(hint(prompts[i]), reference_data_uri(), d, mp, _data_uri(fr) if fr else None)
        return fn

    def mk_api(m):
        def fn(i, d):
            krea_image_keyed(m, hint(prompts[i]), frames.get(i), d)
        return fn

    options = []          # (label, fn, workers, route)
    if krea_connected():
        for m in ladder_models():
            options.append((f"Krea credits · {m}", mk_mcp(m), WORKERS, "mcp"))
    if key_list("KREA_API_KEY"):
        for m in ladder_models():
            options.append((f"Krea API · {m}", mk_api(m), WORKERS, "api"))
    if CF_IMAGE_URL:
        options.append(("Cloudflare AI (free)", lambda i, d: cf_image(prompts[i] + ". " + (STYLE_HINT or CF_STYLE), d), 2, "cf"))
    if not options:
        raise RuntimeError("no image provider: connect Krea, set a Krea API key or a Cloudflare worker URL in Settings")

    def dest(i):
        return os.path.join(imgdir, f"img{i:03d}.png")

    dead = set()          # routes that ran out of credit
    for k, (name, fn, workers, route) in enumerate(options):
        if route in dead:
            continue
        last = not any(o[3] not in dead for o in options[k + 1:])
        try:
            fn(unique[0], dest(unique[0]))
        except OutOfCredit as e:
            dead.add(route)
            status(f"⚠️ {esc(name)}: out of credits ({esc(str(e)[:120])}). Skipping the rest of that route ...")
            if last:
                raise
            continue
        except ModelRejected as e:
            print(f"{name} rejected: {e}", flush=True)
            status(f"⏭️ {esc(name)} can't take reference images ({esc(str(e)[:140])}). Moving to the next model ...")
            if last:
                raise
            continue
        except Exception as e:
            if (isinstance(e, KeyFail) or "402" in str(e)) and route != "cf":
                dead.add(route)
            if last:
                raise
            why = "no API balance" if "402" in str(e) else str(e)[:400]
            print(f"{name} failed: {e}", flush=True)
            status(f"⚠️ {esc(name)} unavailable ({esc(why)}). Trying the next image provider ...")
            continue
        status(f"🎨 Images: {esc(name)}")
        try:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(lambda i: fn(i, dest(i)), unique[1:]))
        except Exception as e:
            if isinstance(e, (OutOfCredit, KeyFail)):
                dead.add(route)
            if last:
                raise
            status(f"⚠️ {esc(name)} failed mid-way ({esc(str(e)[:200])}). Redoing all images with the next one ...")
            continue
        return name
    raise RuntimeError("no image provider worked (every Krea model was skipped or out of credit)")


# --- saved progress per video: the IMAGE/JOB BANK (lets a failed run resume, survives redeploys with a volume) ---
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
LAST_JOB = os.path.join(DATA_DIR, "last_job.txt")
os.makedirs(JOBS_DIR, exist_ok=True)
job_error = [""]


def jread(jd, name):
    try:
        with open(os.path.join(jd, name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def jwrite(jd, name, data):
    tmp = os.path.join(jd, name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(jd, name))


def last_job_dir():
    try:
        with open(LAST_JOB) as f:
            d = os.path.join(JOBS_DIR, f.read().strip())
        return d if os.path.isdir(d) else ""
    except OSError:
        return ""


def reset_job(jd, vid, part=None):
    shutil.rmtree(jd, ignore_errors=True)
    os.makedirs(os.path.join(jd, "audio"), exist_ok=True)
    os.makedirs(os.path.join(jd, "images"), exist_ok=True)
    jwrite(jd, "video.json", {"title": vid["title"], "url": vid["url"], "id": vid["id"], "part": list(part) if part else [1, 1]})


def video_length(video_url):
    """Length of the video in seconds (0 if it can't be found)."""
    try:
        r = http("GET", video_url, timeout=30, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"})
        m = re.search(r'"lengthSeconds":"(\d+)"', r.text)
        if m and int(m.group(1)) > 0:
            return int(m.group(1))
    except Exception as e:
        print(f"length via page failed: {e}", flush=True)
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts(cookies=False, skip_download=True)) as y:
            d = (y.extract_info(video_url, download=False) or {}).get("duration")
        if d:
            return int(d)
    except Exception as e:
        print(f"length via yt-dlp failed: {e}", flush=True)
    if key_list("GEMINI_API_KEY"):
        try:
            out = gemini_call("", [{"file_data": {"file_uri": video_url}},
                                   {"text": "How long is this video in seconds? Reply with only the number."}],
                              max_tokens=20, temperature=0, timeout=300)
            m = re.search(r"\d+", out)
            if m:
                return int(m.group(0))
        except Exception as e:
            print(f"length via Gemini failed: {e}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Review gate: a video made with a backup voice never auto-posts until you watched it and pressed Approve
# ---------------------------------------------------------------------------
def review_reasons(jd):
    out = []
    vo = jread(jd, "voice.json") or {}
    if REVIEW_ON_FALLBACK and vo.get("fallback"):
        out.append(f"a backup voice ({vo.get('name')}) was used because your main voice failed")
    return out


def is_approved(jd):
    return bool(jread(jd, "approved.json"))


def is_viewed(jd):
    return bool(jread(jd, "viewed.json"))


# ---------------------------------------------------------------------------
# Thumbnail (redrawn from the source video's thumbnail) + upload details
# ---------------------------------------------------------------------------
def build_thumbnail(vid, jd):
    out = os.path.join(jd, "thumbnail.jpg")
    if os.path.isfile(out):
        return out
    tdir = os.path.join(jd, "thumb")
    os.makedirs(tdir, exist_ok=True)
    srcp = os.path.join(tdir, "source.jpg")
    got = False
    for name in ("maxresdefault", "hqdefault"):
        r = http("GET", f"https://i.ytimg.com/vi/{vid['id']}/{name}.jpg", timeout=30, tries=2)
        if r.status_code == 200 and len(r.content) > 3000:
            with open(srcp, "wb") as f:
                f.write(r.content)
            got = True
            break
    if not got:
        raise RuntimeError("couldn't get the source thumbnail")
    prompt = (describe_frame(srcp, vid["title"])
              + ". Bold, eye-catching 16:9 YouTube thumbnail composition, one clear focal subject, no text.")
    make_images([0], {0: prompt}, tdir, {0: srcp})
    png = os.path.join(tdir, "img000.png")
    for q in (3, 6, 10):                                   # YouTube wants 2 MB at most
        run_ffmpeg(["ffmpeg", "-y", "-i", png, "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
                    "-q:v", str(q), out], "thumbnail")
        if os.path.getsize(out) < 1_900_000:
            break
    return out


def build_extras(vid, jd, new_text, part=(1, 1)):
    """Upload details + thumbnail. A failure here never kills the build (YouTube just picks its own thumbnail)."""
    if not jread(jd, "meta.json"):
        try:
            status("📝 Writing the title, description and tags ...")
            meta = make_metadata(new_text, vid["title"])
            if part[1] > 1:
                suf = f" (Part {part[0]}/{part[1]})"
                meta["title"] = meta["title"][:100 - len(suf)] + suf
            jwrite(jd, "meta.json", meta)
        except Exception as e:
            status(f"⚠️ Couldn't write the upload details: {esc(str(e)[:160])}")
    if not os.path.isfile(os.path.join(jd, "thumbnail.jpg")):
        try:
            status("🖼️ Redrawing the thumbnail from the source thumbnail ...")
            build_thumbnail(vid, jd)
            status("🖼️ Thumbnail ready.")
        except Exception as e:
            print(f"thumbnail failed: {e}", flush=True)
            status(f"⚠️ Couldn't make the thumbnail: {esc(str(e)[:160])}")


# ---------------------------------------------------------------------------
# Posting to YouTube through Zernio (key pool: if one Zernio key is refused the next is tried)
# ---------------------------------------------------------------------------
ZERNIO = "https://zernio.com/api/v1"
post_lock = threading.Lock()
post_running = False


def zernio_headers():
    key = getattr(_tl, "zernio_key", None) or first_key("ZERNIO_API_KEY")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def zernio_account():
    if ZERNIO_ACCOUNT_ID:
        return ZERNIO_ACCOUNT_ID.strip()
    r = http("GET", f"{ZERNIO}/accounts", headers=zernio_headers(), timeout=30, tries=2)
    key_check(r, "Zernio")
    if r.status_code != 200:
        raise RuntimeError(f"Zernio accounts {r.status_code}: {r.text[:160]}")
    for d in _walk(r.json()):
        if str(d.get("platform", "")).lower() == "youtube":
            acc = d.get("_id") or d.get("id") or d.get("accountId")
            if acc:
                return str(acc)
    raise RuntimeError("no YouTube channel is connected in Zernio (connect it in the Zernio dashboard first)")


def zernio_upload(path, ctype):
    """Presigned upload: returns the public URL Zernio will fetch the file from."""
    name = os.path.basename(path)
    j, err = None, ""
    for body in ({"filename": name, "contentType": ctype}, {"fileName": name, "fileType": ctype}):
        r = http("POST", f"{ZERNIO}/media/presign", headers=zernio_headers(), json=body, timeout=60, tries=1)
        if r.status_code in (401, 403, 429):
            key_check(r, "Zernio")
        if r.status_code in (200, 201):
            j = r.json()
            break
        err = f"{r.status_code} {r.text[:160]}"
    if j is None:
        raise RuntimeError(f"Zernio upload slot failed: {err}")
    up, pub = _find_key(j, ["uploadUrl"]), _find_key(j, ["publicUrl"])
    if not up or not pub:
        raise RuntimeError(f"Zernio gave no upload URL: {str(j)[:160]}")
    check_kill()
    with open(path, "rb") as f:
        r = requests.put(up, data=f, headers={"Content-Type": ctype}, timeout=3600)
    if r.status_code >= 300:
        raise RuntimeError(f"Zernio file upload failed ({r.status_code}): {r.text[:160]}")
    return pub


def _do_post(jd, video, meta, vis):
    status("🚀 Posting to YouTube: finding your channel in Zernio ...")
    acc = zernio_account()
    status("🚀 Uploading the video to Zernio ...")
    video_url = zernio_upload(video, "video/mp4")
    item = {"type": "video", "url": video_url}
    thumb = os.path.join(jd, "thumbnail.jpg")
    if os.path.isfile(thumb):
        status("🚀 Uploading the thumbnail ...")
        try:
            item["thumbnail"] = zernio_upload(thumb, "image/jpeg")
        except KeyFail:
            raise
        except Exception as e:
            status(f"⚠️ Thumbnail upload failed ({esc(str(e)[:120])}) - posting without a custom thumbnail.")
    check_kill()
    body = {"content": meta.get("description") or meta.get("title") or "",
            "tags": meta.get("tags") or [],
            "mediaItems": [item],
            "platforms": [{"platform": "youtube", "accountId": acc,
                           "platformSpecificData": {"title": (meta.get("title") or "Video")[:100],
                                                    "visibility": vis, "categoryId": "27",
                                                    "madeForKids": False,
                                                    "containsSyntheticMedia": bool(MARK_SYNTHETIC)}}],
            "publishNow": True}
    status(f"🚀 Creating the YouTube post ({vis}) ...")
    r = http("POST", f"{ZERNIO}/posts", headers=zernio_headers(), json=body, timeout=600, tries=1)
    if r.status_code in (401, 429) or (r.status_code == 403 and "suspend" not in r.text.lower()):
        key_check(r, "Zernio")
    try:
        j = r.json()
    except ValueError:
        j = {}
    if r.status_code not in (200, 201, 207):
        raise RuntimeError(f"Zernio post {r.status_code}: {r.text[:240]}")
    post = (j.get("post") or j) if isinstance(j, dict) else {}
    plat = next((p for p in (post.get("platforms") or []) if str(p.get("platform")).lower() == "youtube"), {})
    st = str(plat.get("status") or post.get("status") or "").lower()
    if st == "failed" or r.status_code == 207:
        raise RuntimeError(f"YouTube refused it: {plat.get('errorMessage') or j.get('error') or str(j)[:200]}")
    url = plat.get("platformPostUrl") or ""
    jwrite(jd, "posted.json", {"url": url, "status": st, "visibility": vis, "at": int(time.time()),
                               "zernio_post_id": post.get("_id") or post.get("id") or ""})
    status(f'✅ Posted to YouTube ({esc(vis)}, status: {esc(st or "sent")}): '
           + (f'<a href="{esc(url)}" target="_blank">{esc(url)}</a>' if url else "the link appears in your Zernio dashboard when YouTube finishes processing."))
    return jread(jd, "posted.json")


def post_to_youtube(jd=None):
    """Post a finished video. NOT made for kids. Never posts the same job twice, and never posts a video that
    still needs your approval (backup voice used) until you watched it and pressed Approve."""
    check_kill()
    if PAUSED:
        raise RuntimeError("the bot is paused (kill switch)")
    if not key_list("ZERNIO_API_KEY"):
        raise RuntimeError("add your Zernio API key on the Keys tab first")
    jd = jd or last_job_dir()
    video = os.path.join(jd, "final.mp4") if jd else ""
    if not jd or not jread(jd, "done.json") or not os.path.isfile(video):
        raise RuntimeError("there is no finished video to post yet")
    already = jread(jd, "posted.json")
    if already:
        status(f'ℹ️ Already posted: <a href="{esc(already.get("url"))}" target="_blank">{esc(already.get("url") or "see Zernio")}</a> '
               '(use "Start fresh" on the same video to build and post a new one).')
        return already
    reasons = review_reasons(jd)
    if reasons and not is_approved(jd):
        raise RuntimeError("needs your approval first: " + "; ".join(reasons) +
                           ". Watch it on the Results tab, press Approve, then post.")
    meta = jread(jd, "meta.json")
    if not meta:
        meta = make_metadata((jread(jd, "rewrite.json") or {}).get("text") or [], (jread(jd, "video.json") or {}).get("title", ""))
        jwrite(jd, "meta.json", meta)
    vis = POST_VISIBILITY.strip().lower()
    vis = vis if vis in ("private", "unlisted", "public") else "private"

    def go(key):
        _tl.zernio_key = key
        try:
            return _do_post(jd, video, meta, vis)
        finally:
            _tl.zernio_key = None

    return with_keys("ZERNIO_API_KEY", go)


def run_post():
    global post_running
    if PAUSED:
        status("⛔ The bot is paused - press Resume first.")
        return
    if job_running:
        status("⏳ Wait for the build to finish before posting.")
        return
    if not post_lock.acquire(blocking=False):
        status("⏳ A post is already in progress.")
        return
    post_running = True
    try:
        post_to_youtube()
    except Killed:
        status("⛔ Posting stopped by the kill switch.")
    except Exception as e:
        print(f"Post failed: {type(e).__name__}: {e}", flush=True)
        status(f"❌ Posting failed: {esc(str(e)[:400])}")
    finally:
        post_running = False
        post_lock.release()


# ---------------------------------------------------------------------------
# Scenes (watching the video), splitting in two, and the per-video build
# ---------------------------------------------------------------------------
def stage_scenes(vid, seconds, bjd, sc, work):
    """Returns (scenes, visuals, owner, frames, src). Uses the saved scenes when they exist."""
    src = None
    if sc:
        scenes, visuals, owner = sc["scenes"], sc.get("visuals"), sc["owner"]
        status(f"♻️ Reusing the {len(scenes)} scenes saved from the last run (tick 'Start fresh' to redo everything).")
        return scenes, visuals, owner, [None] * len(scenes), src
    scenes = visuals = owner = frames = None
    if key_list("GEMINI_API_KEY") and os.environ.get("WATCH_WITH_GEMINI", "1") != "0":
        status("👀 Asking Gemini to watch the video (it opens the YouTube link itself - no download needed) ...")
        try:
            scenes, visuals, owner = gemini_watch(vid["url"], seconds)
            frames = [None] * len(scenes)
            status(f"🎬 Gemini watched it: {len(scenes)} scenes, {len(set(owner))} different looks")
        except Exception as e:
            print(f"Gemini watch failed: {e}", flush=True)
            scenes = None
            status(f"⚠️ Gemini couldn't watch it ({esc(str(e)[:200])}). Trying the downloader instead ...")

    if scenes is None:
        fetch_cookies_from_github()
        status("📝 Getting subtitles ...")
        cues = yt_call(get_subtitles, vid["url"], work)
        status(f"📝 {len(cues)} subtitle lines")

        status("⬇️ Getting the video to watch it ...")
        src = get_source_video(vid, seconds, work)
        if not src:
            status("⚠️ No way to download the video, so the bot can't watch it. "
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

        frames, owner, hashes = [], [], []
        for i, (a, b, _) in enumerate(scenes):
            check_kill()
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
        status(f"🎬 {len(scenes)} scenes, {len(set(owner))} different looks"
               + (" (reusing images when the video didn't change)" if src else ""))
    jwrite(bjd, "scenes.json", {"seconds": seconds, "scenes": scenes, "visuals": visuals, "owner": owner})
    return scenes, visuals, owner, frames, src


def split_point(scenes):
    """Index of the scene that starts closest to the middle of the video (both halves keep at least one scene)."""
    target = (scenes[0][0] + scenes[-1][1]) / 2
    return min(range(1, len(scenes)), key=lambda i: abs(scenes[i][0] - target))


def build_job(vid, jd, scenes, visuals, owner, frames, src, seconds, work, part, fresh):
    """Stages 3-7 for ONE video (or one half of a split video). Returns the job dir."""
    p, pn = part
    plabel = f" (part {p}/{pn})" if pn > 1 else ""
    old = None if fresh else jread(jd, "scenes.json")
    if old and old.get("scenes") == scenes and old.get("seconds") == seconds:
        jwrite(jd, "video.json", {"title": vid["title"], "url": vid["url"], "id": vid["id"], "part": [p, pn]})
    else:
        reset_job(jd, vid, part)
        jwrite(jd, "scenes.json", {"seconds": seconds, "scenes": scenes, "visuals": visuals, "owner": owner})
    for stale in ("done.json", "approved.json", "viewed.json", "probe.json"):
        try:
            os.remove(os.path.join(jd, stale))
        except OSError:
            pass
    with open(LAST_JOB, "w") as f:
        f.write(os.path.basename(jd))
    n = len(scenes)
    unique = sorted(set(owner))
    if pn > 1:
        status(f"🧩 Building part {p} of {pn} ({n} scenes) ...")

    # ---- stage 3: rewrite (Groq) ----
    check_kill()
    rw = jread(jd, "rewrite.json")
    if rw and len(rw.get("text") or []) == n:
        new_text = rw["text"]
        status("♻️ Reusing the saved rewrite.")
    else:
        status("✍️ Rewriting the narration with " + ("Groq" if (USE_GROQ and key_list("GROQ_API_KEY")) else "Gemini/Anthropic")
               + " (same meaning, new structure) ...")
        new_text = rewrite_scenes(scenes, vid["title"])
        for stale in ("voice.json", "images.json"):
            try:
                os.remove(os.path.join(jd, stale))
            except OSError:
                pass
        jwrite(jd, "rewrite.json", {"text": new_text})

    # ---- stage 4: voice-over ----
    check_kill()
    audios = [os.path.join(jd, "audio", f"a{i:03d}.mp3") for i in range(n)]
    vo = jread(jd, "voice.json")
    if vo and vo.get("sig", voice_sig()) == voice_sig() and all(os.path.isfile(a) and os.path.getsize(a) > 500 for a in audios):
        voice_used = vo.get("name", "")
        status(f"♻️ Reusing the saved voice-over ({esc(voice_used)}).")
    else:
        status("🎙️ Generating voice-over ...")
        voice_used, backup = make_voiceovers(new_text, audios)
        jwrite(jd, "voice.json", {"name": voice_used, "sig": voice_sig(), "fallback": bool(backup)})
        if backup:
            status(f"🛑 A backup voice ({esc(voice_used)}) had to be used. The video will be built, but it will NOT auto-post: "
                   "watch it and press Approve first.")

    # ---- stage 5: images (style reference + real frame) ----
    check_kill()
    imgdir = os.path.join(jd, "images")
    os.makedirs(imgdir, exist_ok=True)
    images = {i: os.path.join(imgdir, f"img{i:03d}.png") for i in unique}
    im = jread(jd, "images.json")
    if im and all(os.path.isfile(pth) for pth in images.values()):
        prompts = {int(k): v for k, v in (im.get("prompts") or {}).items()}
        provider = im.get("provider", "")
        status(f"♻️ Reusing the saved images from the bank ({esc(provider)}).")
    else:
        for fn_ in os.listdir(imgdir):
            try:
                os.remove(os.path.join(imgdir, fn_))
            except OSError:
                pass
        frame_map = {}
        if FRAME_REFS:
            fsrc = src or get_source_video(vid, seconds, work)
            if fsrc:
                status("🖼️ Cutting a real frame for every scene (sent to Krea as a second reference) ...")
                frame_map = make_frames(fsrc, scenes, unique, os.path.join(jd, "frames"))
            else:
                status("ℹ️ No source video available - drawing from the text + your style image only.")
        status(f"🎨 Redrawing {len(unique)} frames ...")
        prompts = {}

        def get_prompt(i):
            check_kill()
            prompts[i] = visuals[i] if visuals and visuals[i] else describe_frame(frame_map.get(i) or frames[i], scenes[i][2])

        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(get_prompt, unique))
        provider = make_images(unique, prompts, imgdir, frame_map)
        if not provider:
            raise RuntimeError("no image provider worked")
        jwrite(jd, "images.json", {"provider": provider, "prompts": {str(k): v for k, v in prompts.items()},
                                   "frames": len(frame_map)})

    # ---- stage 6: final video (verified, so it plays on any phone / browser) ----
    status("🎞️ Animating scenes and joining them ...")
    parts = []
    for i in range(n):
        check_kill()
        dst = os.path.join(work, f"s{i:03d}_{p}.mp4")
        scene_clip(images[owner[i]], audios[i], i, dst)
        parts.append(dst)
    final = os.path.join(work, f"final_{p}.mp4")
    concat(parts, final, work)
    ok, note = verify_video(final)
    if not ok:
        status(f"⚠️ The joined video didn't pass the playback check ({esc(note)}). Re-encoding it ...")
        concat_reencode(parts, final, work)
        ok, note = verify_video(final)
    jwrite(jd, "probe.json", {"ok": ok, "note": note})
    status(("✅ Playback check passed: " if ok else "⚠️ Playback check FAILED: ") + esc(note))
    shutil.move(final, os.path.join(jd, "final.mp4"))

    lines = [f"SOURCE: {vid['title']}  ({vid['url']}){plabel}", f"VOICE: {voice_used}", f"IMAGES: {provider}", ""]
    for i, (a, b, t) in enumerate(scenes):
        lines += [f"--- Scene {i + 1}  [{a:.1f}s-{b:.1f}s]  image from scene {owner[i] + 1}",
                  f"ORIGINAL : {t}", f"REWRITTEN: {new_text[i]}",
                  f"IMAGE PROMPT: {prompts.get(owner[i], '')}", ""]
    with open(os.path.join(jd, "script.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    jwrite(jd, "done.json", {"voice": voice_used, "images": provider})

    mb = os.path.getsize(os.path.join(jd, "final.mp4")) / 1_000_000
    status(f'✅ Video{esc(plabel)} ready ({mb:.1f} MB) - see the Results tab.')

    # ---- stage 7: thumbnail + upload details ----
    check_kill()
    build_extras(vid, jd, new_text, part)
    return jd


def build_preview(handle, seconds=None, fresh=False, whole=False):
    """Builds one video (or two halves). Returns the list of finished job dirs."""
    check_setup()
    seconds = max(20, int(seconds or PREVIEW_SECONDS))
    work = tempfile.mkdtemp(prefix="fl_")
    job_error[0] = ""
    try:
        status(f"🔎 Looking up {esc(handle)} ...")
        vid = yt_call(latest_video, handle)
        status(f"📺 Video: {esc(vid['title'][:90])}")
        if whole:
            status("📏 Checking how long the video is ...")
            length = video_length(vid["url"])
            if length <= 0:
                status(f"⚠️ Couldn't read the video length - using the first {seconds}s instead.")
            else:
                if length > WHOLE_MAX:
                    status(f"⚠️ The video is {length // 60} min; analysing only the first {WHOLE_MAX // 60} min (WHOLE_MAX_SECONDS).")
                    length = WHOLE_MAX
                seconds = int(length)
                status(f"📏 Analysing the whole video ({seconds // 60}:{seconds % 60:02d}).")

        vkey = re.sub(r"[^\w-]", "", vid["id"]) or "video"
        want_split = bool(SPLIT_IN_TWO)
        bjd = os.path.join(JOBS_DIR, vkey + ("_src" if want_split else ""))
        os.makedirs(JOBS_DIR, exist_ok=True)
        sc = None if fresh else jread(bjd, "scenes.json")
        if sc and sc.get("seconds") == seconds and sc.get("scenes") and sc.get("owner"):
            jwrite(bjd, "video.json", {"title": vid["title"], "url": vid["url"], "id": vid["id"], "part": [1, 1]})
        else:
            sc = None
            reset_job(bjd, vid)

        scenes, visuals, owner, frames, src = stage_scenes(vid, seconds, bjd, sc, work)

        splits = [(0, len(scenes))]
        if want_split:
            if len(scenes) >= 2:
                cut = split_point(scenes)
                splits = [(0, cut), (cut, len(scenes))]
                status(f"✂️ Splitting into 2 videos: part 1 = scenes 1-{cut}, part 2 = scenes {cut + 1}-{len(scenes)}.")
            else:
                status("ℹ️ Only one scene found, so it can't be split - building one video.")

        outs = []
        for pi, (lo, hi) in enumerate(splits):
            check_kill()
            pn = len(splits)
            name = f"{vkey}-p{pi + 1}" if pn > 1 else vkey
            s_scenes = scenes[lo:hi]
            s_vis = visuals[lo:hi] if visuals else None
            s_owner = [(o - lo) if o >= lo else 0 for o in owner[lo:hi]]
            s_frames = frames[lo:hi] if frames else [None] * (hi - lo)
            outs.append(build_job(vid, os.path.join(JOBS_DIR, name), s_scenes, s_vis, s_owner, s_frames, src,
                                  seconds, work, (pi + 1, pn), fresh))
        return outs
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------
job_lock = threading.Lock()
job_running = False


def busy():
    return job_running or post_running


def run_job(handle, seconds, fresh=False, whole=False):
    global job_running
    if PAUSED:
        status("⛔ The bot is PAUSED (kill switch). Press Resume first.")
        return
    if not job_lock.acquire(blocking=False):
        status("⏳ A preview is already being built.")
        return
    job_running = True
    try:
        outs = build_preview(handle, seconds, fresh, whole)
        if AUTO_POST:
            status("🚀 Auto-post is ON.")
            for jd in outs:
                title = (jread(jd, "video.json") or {}).get("title", "")[:50]
                reasons = review_reasons(jd)
                if reasons and not is_approved(jd):
                    status(f"🛑 Auto-post SKIPPED for “{esc(title)}”: {esc('; '.join(reasons))}. "
                           "Watch it on the Results tab, press Approve, then press Post.")
                    continue
                try:
                    post_to_youtube(jd)
                except Killed:
                    raise
                except Exception as e:
                    print(f"Auto-post failed: {type(e).__name__}: {e}", flush=True)
                    status(f"❌ Built, but posting failed: {esc(str(e)[:400])}. You can press 'Post' on the Results tab to retry.")
    except Killed:
        job_error[0] = "stopped by the kill switch"
        status("⛔ Stopped by the kill switch. Everything made so far is saved - check it on the Results tab.")
    except Exception as e:
        print(f"Job failed: {type(e).__name__}: {e}", flush=True)
        job_error[0] = str(e)[:300] or "failed"
        if "Sign in to confirm" in str(e):
            have = "Your cookies were rejected or expired - export fresh ones" if cookies_path() else "Upload a YouTube cookies.txt on the Settings tab (or set YOUTUBE_COOKIES_B64)"
            status(f"❌ YouTube thinks the server is a bot. {have}.")
        else:
            status(f"❌ {esc(str(e)[:400])}")
    finally:
        job_running = False
        job_lock.release()


# ---------------------------------------------------------------------------
# Dashboard (v3): sticky header with the kill switch, tabs, live pipeline, key manager
# ---------------------------------------------------------------------------
PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Faceless Bot</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0a0b10"><style>
:root{--bg:#0a0b10;--s1:#12141c;--s2:#1a1d28;--s3:#222636;--bd:#272b3c;--tx:#eceef6;--mu:#8b90a8;--a:#ff4d6d;--a2:#7c5cff;
--ok:#3ddc97;--warn:#ffb454;--bad:#ff5a5f;--r:14px}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;
padding-bottom:env(safe-area-inset-bottom)}
a{color:#9d8cff;text-decoration:none}a:hover{text-decoration:underline}code{background:var(--s3);padding:1px 6px;border-radius:6px;font-size:12.5px}
header.top{position:sticky;top:0;z-index:50;background:rgba(10,11,16,.92);backdrop-filter:blur(10px);border-bottom:1px solid var(--bd);
display:flex;align-items:center;gap:12px;padding:10px 16px;padding-top:calc(10px + env(safe-area-inset-top))}
.brand{font-weight:800;letter-spacing:.2px;font-size:16px;display:flex;align-items:center;gap:8px}
.logo{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,var(--a),var(--a2));display:grid;place-items:center;font-size:14px}
.sp{flex:1}.pill{font-size:12px;font-weight:700;padding:5px 11px;border-radius:999px;background:var(--s2);border:1px solid var(--bd);white-space:nowrap}
.pill.ok{color:var(--ok)}.pill.bad{color:var(--bad);border-color:rgba(255,90,95,.45)}.pill.work{color:var(--warn)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--warn);margin-right:6px;animation:pulse 1.1s infinite}
@keyframes pulse{50%{opacity:.25}}
#killbox form{margin:0}
button,.btn{font:inherit;font-weight:700;font-size:14px;border:0;border-radius:11px;padding:11px 18px;cursor:pointer;color:#fff;
background:linear-gradient(135deg,var(--a),var(--a2));transition:transform .08s,opacity .15s;display:inline-block}
.btn:hover{text-decoration:none}
button:active{transform:scale(.97)}button:disabled{opacity:.4;cursor:not-allowed}
button.kill{background:#d4001e;padding:9px 14px;font-size:13px;letter-spacing:.3px;box-shadow:0 0 0 1px rgba(255,255,255,.1) inset}
button.ghost{background:var(--s3);color:var(--tx)}button.small{padding:6px 11px;font-size:12px;border-radius:9px}
button.danger{background:#3a1218;color:#ff8c90}button.good{background:linear-gradient(135deg,#1fb97a,#3ddc97);color:#04140d}
.ticker{font-size:12.5px;color:var(--mu);padding:7px 16px;border-bottom:1px solid var(--bd);background:var(--s1);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ticker a{color:#9d8cff}
.alerts{padding:0 16px}.alert{margin:12px 0 0;padding:12px 14px;border-radius:var(--r);border:1px solid var(--bd);background:var(--s1);font-size:13.5px}
.alert.bad{border-color:rgba(255,90,95,.5);background:rgba(255,90,95,.07)}.alert.warn{border-color:rgba(255,180,84,.5);background:rgba(255,180,84,.07)}
.alert form{display:flex;gap:8px;margin-top:9px}.alert input{flex:1}
nav.tabs{position:sticky;top:53px;z-index:40;display:flex;gap:4px;overflow-x:auto;padding:8px 16px;background:var(--bg);border-bottom:1px solid var(--bd);scrollbar-width:none}
nav.tabs::-webkit-scrollbar{display:none}
nav.tabs button{background:transparent;color:var(--mu);padding:8px 15px;border-radius:999px;font-weight:600;white-space:nowrap}
nav.tabs button.on{background:var(--s3);color:var(--tx)}
main{max-width:920px;margin:0 auto;padding:16px}
.tab{display:none}.tab.on{display:block;animation:fade .15s}@keyframes fade{from{opacity:.4}}
.card{background:var(--s1);border:1px solid var(--bd);border-radius:var(--r);padding:18px;margin-bottom:14px}
.card h2,.card h3{margin:0 0 12px;font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--mu);font-weight:700}
.card.ok{border-color:rgba(61,220,151,.4)}.card.warn{border-color:rgba(255,180,84,.45)}.card.bad{border-color:rgba(255,90,95,.45)}
.hint,.m{color:var(--mu);font-size:12.5px}.hint{margin-top:8px;line-height:1.5}
label.f{display:block;font-size:12.5px;color:var(--mu);margin:12px 0 6px;font-weight:600}
input[type=text],input[type=number],input[type=password],select,textarea{width:100%;background:var(--s2);border:1px solid var(--bd);color:var(--tx);
border-radius:10px;padding:11px 12px;font:inherit}input:focus,select:focus{outline:2px solid rgba(124,92,255,.5);border-color:transparent}
.seg{display:flex;background:var(--s2);border:1px solid var(--bd);border-radius:12px;padding:4px;gap:4px;margin-bottom:12px}
.seg label{flex:1;text-align:center;padding:9px 6px;border-radius:9px;cursor:pointer;font-weight:600;font-size:13.5px;color:var(--mu)}
.seg input{display:none}.seg input:checked+span{color:#fff}.seg label:has(input:checked){background:var(--s3);color:#fff}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:600px){.grid2{grid-template-columns:1fr}}
.sw{display:flex;align-items:center;gap:12px;padding:10px 12px;background:var(--s2);border:1px solid var(--bd);border-radius:12px;margin-top:10px;cursor:pointer}
.sw input{display:none}.track{width:42px;height:24px;border-radius:99px;background:#3a3f55;position:relative;flex:none;transition:.15s}
.knob{position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;background:#fff;transition:.15s}
.sw input:checked+.track{background:linear-gradient(135deg,var(--a),var(--a2))}.sw input:checked+.track .knob{left:21px}
.swt{font-size:13.5px;line-height:1.35}.swt small{display:block;color:var(--mu);font-size:12px}
.bigbtn{width:100%;margin-top:16px;padding:15px;font-size:16px}
.steps{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px;scrollbar-width:none}.steps::-webkit-scrollbar{display:none}
.step{flex:1;min-width:74px;text-align:center;background:var(--s2);border:1px solid var(--bd);border-radius:11px;padding:9px 4px;font-size:11.5px;color:var(--mu)}
.step b{display:block;font-size:17px;line-height:1.2}.step.done{color:var(--ok);border-color:rgba(61,220,151,.35)}
.step.act{color:var(--warn);border-color:rgba(255,180,84,.6);animation:pulse 1.2s infinite}.step.err{color:var(--bad);border-color:rgba(255,90,95,.6)}
.bar{height:6px;background:var(--s3);border-radius:99px;margin:12px 0 0;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--a),var(--a2));transition:width .4s}
.feed{display:flex;flex-direction:column;gap:6px;max-height:340px;overflow-y:auto}
.fi{background:#0a0c12;border:1px solid var(--bd);border-radius:10px;padding:8px 11px;font-size:13px}
video{width:100%;border-radius:12px;background:#000;max-height:70vh}
.acts{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}.acts .btn,.acts button{padding:9px 14px;font-size:13px}
.btn.ghost{background:var(--s3);color:var(--tx);text-decoration:none;display:inline-block;border-radius:11px}
.chip{display:inline-block;font-size:11.5px;font-weight:700;padding:3px 9px;border-radius:99px;background:var(--s3);color:var(--mu);margin:0 4px 4px 0}
.chip.ok{color:var(--ok)}.chip.warn{color:var(--warn)}.chip.bad{color:var(--bad)}
details{background:var(--s2);border:1px solid var(--bd);border-radius:12px;margin-bottom:8px}
summary{padding:12px 14px;cursor:pointer;font-weight:600;font-size:14px}.sb{padding:4px 14px 14px;font-size:13px}
.sc{border-top:1px solid var(--bd);padding:9px 0}.rw{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px}.rw>div{background:#0a0c12;border-radius:8px;padding:8px}
@media(max-width:600px){.rw{grid-template-columns:1fr}}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}.gal img{width:100%;border-radius:9px;display:block}
img.th{width:100%;max-width:360px;border-radius:10px;border:1px solid var(--bd);display:block;margin-bottom:8px}
audio{width:100%;height:34px;margin-top:4px}
.kp-h{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}.kp-h b{font-size:15px}
.krow{display:flex;align-items:center;gap:10px;padding:9px 0;border-top:1px solid var(--bd);flex-wrap:wrap;font-size:13px}
.krow code{font-size:13px}.krow .st{flex:1;min-width:140px}.st.ok{color:var(--ok)}.st.bad{color:var(--bad)}.st.idle{color:var(--mu)}
.krow form{margin:0}.addkey{display:flex;gap:8px;margin-top:10px}
.jobs .job{display:flex;gap:12px;align-items:flex-start;padding:12px 0;border-top:1px solid var(--bd)}
.jobs .job img{width:112px;border-radius:8px;flex:none}.jobs .job .nt{width:112px;height:63px;border-radius:8px;background:var(--s3);flex:none}
.job h4{margin:0 0 4px;font-size:14px}.job form{display:inline}
.empty{text-align:center;color:var(--mu);padding:34px 10px}
.upl{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.ref{max-height:130px;border-radius:10px;border:1px solid var(--bd)}
.stick{position:sticky;bottom:10px;z-index:5}
</style></head><body>
<header class="top"><div class="brand"><span class="logo">▶</span><span>Faceless Bot</span></div><div class="sp"></div>
<span id="pill">@@PILL@@</span><div id="killbox">@@KILL@@</div></header>
<div class="ticker" id="ticker">@@TICKER@@</div>
<div class="alerts" id="alert">@@ALERT@@</div>
<nav class="tabs"><button data-tab="build" onclick="tab('build')">Build</button><button data-tab="results" onclick="tab('results')">Results</button>
<button data-tab="keys" onclick="tab('keys')">Keys</button><button data-tab="settings" onclick="tab('settings')">Settings</button>
<button data-tab="bank" onclick="tab('bank')">Bank</button></nav>
<main>
<section class="tab" id="tab-build">@@BUILD@@</section>
<section class="tab" id="tab-results"><div id="results">@@RESULTS@@</div></section>
<section class="tab" id="tab-keys">@@KEYS@@</section>
<section class="tab" id="tab-settings">@@SETTINGS@@</section>
<section class="tab" id="tab-bank">@@BANK@@</section>
</main>
<script>
window.RUNNING=@@RUNNING@@;
const TABS=['build','results','keys','settings','bank'];
function tab(n){if(!TABS.includes(n))n='build';document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id==='tab-'+n));
document.querySelectorAll('nav.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.tab===n));
try{localStorage.setItem('tab',n)}catch(e){}history.replaceState(null,'','#'+n);window.scrollTo(0,0)}
tab((location.hash||'').slice(1)||(function(){try{return localStorage.getItem('tab')}catch(e){return 'build'}})()||'build');
function setHTML(sel,html,keep){const el=document.querySelector(sel);if(!el||html===undefined)return;
if(el.dataset.h===html)return;el.dataset.h=html;
const open=keep?[...el.querySelectorAll('details[open]')].map(d=>d.dataset.id):[];el.innerHTML=html;
open.forEach(id=>{const d=el.querySelector('details[data-id="'+id+'"]');if(d)d.open=true})}
async function poll(){try{const r=await fetch('/?partial=1',{cache:'no-store'});const j=await r.json();
setHTML('#pill',j.pill);setHTML('#killbox',j.kill);setHTML('#ticker',j.ticker);setHTML('#alert',j.alert);
setHTML('#progress',j.progress);setHTML('#activity',j.activity);
const v=document.querySelector('#results video');if(!(v&&!v.paused))setHTML('#results',j.results,true);
const b=document.getElementById('runbtn');if(b){b.disabled=j.run;b.textContent=j.run?'Working...':'Build'}
if(j.run)setTimeout(poll,3000)}catch(e){setTimeout(poll,6000)}}
if(window.RUNNING)poll();
async function up(i){const f=i.files[0];if(!f)return;const m=document.getElementById('upmsg');m.textContent='Uploading...';
const r=await fetch('/upload-ref?name='+encodeURIComponent(f.name),{method:'PUT',body:f});
if(!r.ok){m.textContent='Failed: '+await r.text();return}location.reload()}
async function upc(i){const f=i.files[0];if(!f)return;const m=document.getElementById('upc');m.textContent='Uploading...';
const r=await fetch('/upload-cookies',{method:'PUT',body:f});if(!r.ok){m.textContent='Failed: '+await r.text();return}location.reload()}
document.addEventListener('timeupdate',e=>{const v=e.target;if(!v||v.tagName!=='VIDEO'||!v.dataset.job||v.dataset.sent)return;
if(v.duration&&v.currentTime/v.duration>=0.8){v.dataset.sent=1;
fetch('/viewed',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'id='+encodeURIComponent(v.dataset.job)}).then(()=>{
const b=document.getElementById('approveBtn');if(b){b.disabled=false;b.textContent='✅ Approve this video'}})}},true);
function mode(m){document.getElementById('lbl').textContent=m==='video'?'Link to ONE YouTube video':'Channel handle or link (uses its latest video)';
document.getElementById('handle').placeholder=m==='video'?'https://www.youtube.com/watch?v=...':'@channelname'}
function scope(s){document.getElementById('secs').disabled=(s==='whole')}
</script></body></html>"""


def _mt(path):
    try:
        return int(os.path.getmtime(path))
    except OSError:
        return 0


def _vis():
    v = POST_VISIBILITY.strip().lower()
    return v if v in ("private", "unlisted", "public") else "private"


def _slug(s):
    return re.sub(r"[^\w.-]+", "-", s or "video").strip("-")[:60] or "video"


def job_name(jd):
    return os.path.basename(jd) if jd else ""


def pill_html():
    if PAUSED:
        return '<span class="pill bad">⛔ Paused</span>'
    if job_running:
        return '<span class="pill work"><i class="dot"></i>Building</span>'
    if post_running:
        return '<span class="pill work"><i class="dot"></i>Posting</span>'
    return '<span class="pill ok">● Ready</span>'


def killbox_html():
    if PAUSED:
        return ('<form method="POST" action="/resume" onsubmit="return confirm(\'Resume the bot?\')">'
                '<button class="good" type="submit">▶ Resume</button></form>')
    return '<form method="POST" action="/kill"><button class="kill" type="submit">⛔ KILL</button></form>'


def ticker_html():
    with status_lock:
        return status_feed[-1] if status_feed else "Ready."


def alert_html():
    out = ""
    for name, label, why in keys_alerts():
        out += (f'<div class="alert bad"><b>All {esc(label)} keys are failing</b> - {esc(why[:110])}'
                f'<form method="POST" action="/keys/add"><input type="hidden" name="name" value="{esc(name)}">'
                f'<input type="password" name="key" placeholder="Paste a new {esc(label)} key" autocomplete="off">'
                '<button type="submit">Use it</button></form></div>')
    jd = last_job_dir()
    if jd and jread(jd, "done.json") and review_reasons(jd) and not is_approved(jd) and not jread(jd, "posted.json"):
        out += ('<div class="alert warn">🛑 <b>This video needs your approval before it can post.</b> '
                f'{esc("; ".join(review_reasons(jd)))}. <a href="#results" onclick="tab(\'results\')">Open Results</a></div>')
    if PAUSED:
        out += '<div class="alert bad">⛔ <b>The bot is paused.</b> Nothing builds or posts until you press Resume (top right). What it already made is saved.</div>'
    return out


STEP_LABELS = [("🔎", "Found"), ("👀", "Watched"), ("✍️", "Rewritten"), ("🎙️", "Voice"), ("🎨", "Images"),
               ("🎞️", "Video"), ("🖼️", "Thumb"), ("🚀", "Posted")]


def job_flags(jd):
    vd = jread(jd, "video.json") or {}
    meta = jread(jd, "meta.json")
    return [bool(vd), bool(jread(jd, "scenes.json")), bool(jread(jd, "rewrite.json")), bool(jread(jd, "voice.json")),
            bool(jread(jd, "images.json")), bool(jread(jd, "done.json")),
            bool(meta) or os.path.isfile(os.path.join(jd, "thumbnail.jpg")), bool(jread(jd, "posted.json"))]


def progress_html():
    jd = last_job_dir()
    if not jd:
        return '<div class="empty">No build yet. Paste a link above and press Build.</div>'
    flags = job_flags(jd)
    pend = next((i for i, f in enumerate(flags) if not f), None)
    vd = jread(jd, "video.json") or {}
    part = vd.get("part") or [1, 1]
    cells = ""
    for i, (ic, lb) in enumerate(STEP_LABELS):
        cls = "done" if flags[i] else ("err" if (i == pend and job_error[0]) else ("act" if (i == pend and busy()) else ""))
        cells += f'<div class="step {cls}"><b>{"✅" if flags[i] else ic}</b>{lb}</div>'
    pct = int(100 * sum(flags) / len(flags))
    err = f'<div class="hint" style="color:var(--bad)">Stopped: {esc(job_error[0][:200])}</div>' if job_error[0] else ""
    title = esc((vd.get("title") or "")[:80]) + (f' <span class="chip">part {part[0]}/{part[1]}</span>' if part[1] > 1 else "")
    return f'<div class="m" style="margin-bottom:8px">{title}</div><div class="steps">{cells}</div><div class="bar"><i style="width:{pct}%"></i></div>{err}'


def activity_html():
    with status_lock:
        items = list(status_feed[-25:])
    return "".join(f'<div class="fi">{i}</div>' for i in reversed(items)) or '<div class="hint">Nothing yet.</div>'


def build_html():
    s = settings_load()
    mode_v = s.get("LAST_MODE", "channel")
    whole = s.get("LAST_WHOLE", "1") == "1"
    sw = lambda n, t, h, c: (f'<label class="sw"><input type="checkbox" name="{n}" value="1"{" checked" if c else ""}>'
                             f'<span class="track"><span class="knob"></span></span><span class="swt">{esc(t)}<small>{esc(h)}</small></span></label>')
    opts = "".join(f'<option value="{v}"{" selected" if v == _vis() else ""}>{v}</option>' for v in ("private", "unlisted", "public"))
    return (
        '<div class="card"><h2>New build</h2><form method="POST" action="/run">'
        '<div class="seg"><label><input type="radio" name="mode" value="channel" onchange="mode(this.value)"'
        + (" checked" if mode_v == "channel" else "") + '><span>📺 Channel</span></label>'
        '<label><input type="radio" name="mode" value="video" onchange="mode(this.value)"'
        + (" checked" if mode_v == "video" else "") + '><span>🎯 One video link</span></label></div>'
        f'<label class="f" id="lbl">{"Link to ONE YouTube video" if mode_v == "video" else "Channel handle or link (uses its latest video)"}</label>'
        f'<input type="text" id="handle" name="handle" required autocomplete="off" placeholder="{"https://www.youtube.com/watch?v=..." if mode_v == "video" else "@channelname"}" '
        f'value="{esc(s.get("LAST_HANDLE", ""))}">'
        '<div class="seg" style="margin-top:12px"><label><input type="radio" name="scope" value="whole" onchange="scope(this.value)"'
        + (" checked" if whole else "") + '><span>Whole video</span></label>'
        '<label><input type="radio" name="scope" value="secs" onchange="scope(this.value)"' + ("" if whole else " checked")
        + '><span>First N seconds</span></label></div>'
        f'<input type="number" id="secs" name="seconds" min="20" max="3600" value="{PREVIEW_SECONDS}"{" disabled" if whole else ""}>'
        + sw("split", "Split into two videos", "Cuts the video at the middle scene: Part 1 and Part 2 are built (and posted) separately", SPLIT_IN_TWO)
        + sw("autopost", "Post to YouTube automatically when done", "A video that needed a backup voice is never auto-posted - you approve it first", AUTO_POST)
        + f'<label class="f">Visibility when posting</label><select name="visibility">{opts}</select>'
        + sw("fresh", "Start fresh", "Ignore everything saved for this video and rebuild it all", False)
        + f'<button class="bigbtn" type="submit" id="runbtn"{" disabled" if busy() else ""}>{"Working..." if busy() else "Build"}</button></form>'
        '<div class="hint">If a run fails, build again with the same link: it continues from the saved stages instead of paying twice.</div></div>'
        '<div class="card"><h2>Progress</h2><div id="progress">' + progress_html() + '</div></div>'
        '<div class="card"><h2>Activity</h2><div class="feed" id="activity">' + activity_html() + '</div></div>')


def results_html():
    jd = last_job_dir()
    if not jd:
        return '<div class="card"><div class="empty">Nothing built yet.</div></div>'
    jid = job_name(jd)
    vd = jread(jd, "video.json") or {}
    sc = jread(jd, "scenes.json") or {}
    rw = jread(jd, "rewrite.json") or {}
    vo = jread(jd, "voice.json") or {}
    im = jread(jd, "images.json") or {}
    dn = jread(jd, "done.json") or {}
    meta = jread(jd, "meta.json") or {}
    posted = jread(jd, "posted.json") or {}
    probe = jread(jd, "probe.json") or {}
    part = vd.get("part") or [1, 1]
    thumb = os.path.join(jd, "thumbnail.jpg")
    fv = os.path.join(jd, "final.mp4")
    scenes, visuals, owner = sc.get("scenes") or [], sc.get("visuals") or [], sc.get("owner") or []
    texts = rw.get("text") or []
    prompts = {int(k): v for k, v in (im.get("prompts") or {}).items()}
    reasons = review_reasons(jd)
    approved, viewed = is_approved(jd), is_viewed(jd)

    chips = ""
    if vo:
        chips += f'<span class="chip {"warn" if vo.get("fallback") else "ok"}">🎙️ {esc(vo.get("name"))}{" (backup)" if vo.get("fallback") else ""}</span>'
    if im:
        chips += f'<span class="chip ok">🎨 {esc(im.get("provider"))}</span>'
    if probe:
        chips += f'<span class="chip {"ok" if probe.get("ok") else "bad"}">{"▶ plays everywhere" if probe.get("ok") else "⚠ playback check failed"}</span>'
    if posted:
        chips += '<span class="chip ok">🚀 posted</span>'
    elif reasons and not approved:
        chips += '<span class="chip bad">🛑 needs approval</span>'
    elif approved:
        chips += '<span class="chip ok">✅ approved</span>'

    sib = ""
    if part[1] > 1:
        base = jid.rsplit("-p", 1)[0]
        for k in range(1, part[1] + 1):
            n = f"{base}-p{k}"
            if os.path.isdir(os.path.join(JOBS_DIR, n)) and n != jid:
                sib += (f'<form method="POST" action="/bank/open" style="display:inline"><input type="hidden" name="id" value="{esc(n)}">'
                        f'<button class="ghost small" type="submit">Open part {k}</button></form> ')
    head = (f'<div class="card"><h2>Video{f" - part {part[0]}/{part[1]}" if part[1] > 1 else ""}</h2>'
            f'<div style="font-weight:700;font-size:16px;margin-bottom:6px">{esc(meta.get("title") or vd.get("title") or "")}</div>{chips}{sib}')
    if dn and os.path.isfile(fv):
        slug = _slug(meta.get("title") or vd.get("title"))
        head += (f'<video controls playsinline preload="metadata" data-job="{esc(jid)}" src="/jf/{esc(jid)}/final.mp4?v={_mt(fv)}"'
                 + (f' poster="/jf/{esc(jid)}/thumbnail.jpg?v={_mt(thumb)}"' if os.path.isfile(thumb) else "") + '></video>'
                 f'<div class="acts"><a class="btn ghost" href="/dl/{esc(jid)}/final.mp4" download="{esc(slug)}.mp4">⬇ Download video</a>'
                 + (f'<a class="btn ghost" href="/dl/{esc(jid)}/thumbnail.jpg" download="{esc(slug)}-thumbnail.jpg">⬇ Thumbnail</a>' if os.path.isfile(thumb) else "")
                 + f'<a class="btn ghost" href="/dl/{esc(jid)}/script.txt" download="{esc(slug)}-script.txt">⬇ Script</a></div>'
                 + (f'<div class="hint">File check: {esc(probe.get("note"))}. '
                    'If it still won\'t open on your device, tell me the device/app and the exact error.</div>' if probe else ""))
    else:
        head += '<div class="empty">The video isn\'t finished yet.</div>'
    head += '</div>'

    gate = ""
    if dn and reasons and not posted:
        if approved:
            gate = '<div class="card ok"><h2>Review</h2>✅ You approved this video. It can be posted now.</div>'
        else:
            gate = ('<div class="card warn"><h2>Needs your approval</h2>'
                    f'<div>{esc("; ".join(reasons))}.</div>'
                    '<div class="hint">It will never auto-post like this. Watch it (at least 80%) and approve it, then press Post.</div>'
                    f'<form method="POST" action="/approve"><input type="hidden" name="id" value="{esc(jid)}">'
                    f'<button id="approveBtn" class="good" type="submit"{"" if viewed else " disabled"}>'
                    f'{"✅ Approve this video" if viewed else "Watch the video first"}</button></form></div>')

    post = ""
    if posted:
        post = ('<div class="card ok"><h2>YouTube</h2>Posted (' + esc(posted.get("visibility")) + ", " + esc(posted.get("status") or "sent") + "): "
                + (f'<a href="{esc(posted.get("url"))}" target="_blank">{esc(posted.get("url"))}</a>' if posted.get("url") else "check your Zernio dashboard") + "</div>")
    elif dn:
        blocked = bool(reasons) and not approved
        post = ('<div class="card"><h2>YouTube</h2>'
                '<form method="POST" action="/post" onsubmit="return confirm(\'Post this video to YouTube now?\')">'
                f'<input type="hidden" name="id" value="{esc(jid)}"><button type="submit"{" disabled" if blocked else ""}>🚀 Post to YouTube ({_vis()}, not made for kids)</button></form>'
                + ('<div class="hint">Locked until you approve it above.</div>' if blocked else
                   '<div class="hint">Sent through Zernio with the thumbnail and the title / description below.</div>') + '</div>')

    d = ""
    if meta or os.path.isfile(thumb):
        inner = (f'<img class="th" src="/jf/{esc(jid)}/thumbnail.jpg?v={_mt(thumb)}">' if os.path.isfile(thumb) else "")
        inner += (f'<div><b>{esc(meta.get("title"))}</b></div><div class="m" style="white-space:pre-wrap">{esc(meta.get("description"))}</div>'
                  f'<div class="m">Tags: {esc(", ".join(meta.get("tags") or []))}</div>') if meta else ""
        d += f'<details data-id="meta"><summary>🖼️ Thumbnail + title / description</summary><div class="sb">{inner}</div></details>'
    if scenes:
        b = "".join(f'<div class="sc"><b>Scene {i + 1}</b> <span class="m">{a:.1f}s-{c:.1f}s</span><div>{esc(t)}</div></div>'
                    for i, (a, c, t) in enumerate(scenes))
        d += f'<details data-id="scenes"><summary>👀 Scenes ({len(scenes)})</summary><div class="sb">{b}</div></details>'
    if texts:
        b = "".join(f'<div class="sc"><b>Scene {i + 1}</b><div class="rw"><div><span class="m">Original</span><br>{esc(t)}</div>'
                    f'<div><span class="m">Rewritten</span><br>{esc(texts[i])}</div></div></div>'
                    for i, (a, c, t) in enumerate(scenes) if i < len(texts))
        d += f'<details data-id="rewrite"><summary>✍️ Original vs rewritten</summary><div class="sb">{b}</div></details>'
    if vo:
        b = ""
        for i in range(len(texts)):
            ap = os.path.join(jd, "audio", f"a{i:03d}.mp3")
            if os.path.isfile(ap):
                b += (f'<div class="sc"><b>Scene {i + 1}</b><div>{esc(texts[i])}</div>'
                      f'<audio controls preload="none" src="/job/audio/a{i:03d}.mp3?v={_mt(ap)}"></audio></div>')
        d += f'<details data-id="voice"><summary>🎙️ Voice-over ({esc(vo.get("name"))})</summary><div class="sb">{b}</div></details>'
    if im:
        b = f'<div class="m">Drawn with {esc(im.get("provider"))} · real frames used: {esc(im.get("frames", 0))}</div><div class="gal">'
        for i in sorted(set(owner)):
            ip = os.path.join(jd, "images", f"img{i:03d}.png")
            if os.path.isfile(ip):
                u = f"/job/images/img{i:03d}.png?v={_mt(ip)}"
                used = ", ".join(str(k + 1) for k, o in enumerate(owner) if o == i)
                b += (f'<div><a href="{u}" target="_blank"><img loading="lazy" src="{u}"></a>'
                      f'<div class="m">Scene {esc(used)}: {esc(prompts.get(i, ""))}</div></div>')
        b += "</div>"
        d += f'<details data-id="images"><summary>🎨 Images</summary><div class="sb">{b}</div></details>'
    return head + gate + post + (f'<div class="card"><h2>Details</h2>{d}</div>' if d else "")


def keys_html():
    out = ('<div class="card"><h2>How keys work</h2><div class="hint" style="margin:0">Every provider can have <b>several</b> keys. '
           'Keys you paste here and keys from Railway variables (<code>GROQ_API_KEY</code>, <code>GROQ_API_KEY_2</code>, <code>GROQ_API_KEY_3</code> ...) '
           'are pooled. If a key is invalid, out of credit or rate-limited, the bot moves to the next one by itself. '
           'When a whole provider is failing you get a red banner with a box to paste a new key - no redeploy needed. '
           'Newest website key is tried first. Keys are stored in plain text in your data folder, so set a <code>DASHBOARD_PASSWORD</code>.</div></div>')
    for name, label, desc in KEY_PROVIDERS:
        rows = key_rows(name)
        env = " / ".join(KEY_ENV_ALIASES.get(name, [name]))
        rhtml = ""
        for r in rows:
            rm = ""
            if r["removable"]:
                rm = (f'<form method="POST" action="/keys/remove" onsubmit="return confirm(\'Remove this key?\')">'
                      f'<input type="hidden" name="name" value="{esc(name)}"><input type="hidden" name="kid" value="{esc(r["kid"])}">'
                      '<button class="danger small" type="submit">Remove</button></form>')
            rhtml += (f'<div class="krow"><code>{esc(r["mask"])}</code><span class="chip">{esc(r["source"])}</span>'
                      f'<span class="st {r["cls"]}">{esc(r["status"])}</span>{rm}</div>')
        retry = ""
        if any(r["cls"] == "bad" for r in rows):
            retry = (f'<form method="POST" action="/keys/retry" style="display:inline"><input type="hidden" name="name" value="{esc(name)}">'
                     '<button class="ghost small" type="submit">Retry failed keys</button></form>')
        out += (f'<div class="card"><div class="kp-h"><b>{esc(label)}</b><span class="chip {"ok" if rows else "bad"}">{len(rows)} key{"s" if len(rows) != 1 else ""}</span>{retry}</div>'
                f'<div class="m">{esc(desc)}</div>{rhtml or "<div class=hint>No key yet.</div>"}'
                f'<form method="POST" action="/keys/add" class="addkey"><input type="hidden" name="name" value="{esc(name)}">'
                f'<input type="password" name="key" placeholder="Paste a {esc(label)} key" autocomplete="off"><button type="submit">Add</button></form>'
                f'<div class="hint">Railway names: <code>{esc(env)}</code>, with _2, _3 ... for more.</div></div>')
    return out


def settings_html():
    fields = {k: (lb, kd) for k, lb, kd in SETTING_FIELDS}
    form = ""
    for gname, keys in SETTING_GROUPS:
        body = ""
        for k in keys:
            lb, kd = fields[k]
            cur = globals()[k]
            if kd == "bool":
                body += (f'<label class="sw"><input type="checkbox" name="{k}" value="1"{" checked" if cur else ""}>'
                         f'<span class="track"><span class="knob"></span></span><span class="swt">{esc(lb)}</span></label>')
            elif kd == "secret":
                ph = ("saved ····" + str(cur)[-4:]) if cur else "not set"
                body += (f'<label class="f">{esc(lb)} <span class="m">(blank = keep, "-" = clear)</span></label>'
                         f'<input type="password" name="{k}" placeholder="{esc(ph)}" autocomplete="off">')
            elif k == "POST_VISIBILITY":
                o = "".join(f'<option value="{v}"{" selected" if v == _vis() else ""}>{v}</option>' for v in ("private", "unlisted", "public"))
                body += f'<label class="f">{esc(lb)}</label><select name="{k}">{o}</select>'
            else:
                body += f'<label class="f">{esc(lb)}</label><input type="text" name="{k}" value="{esc(cur)}">'
        form += f'<div class="card"><h2>{esc(gname)}</h2>{body}</div>'
    ref = (f'<img class="ref" src="/ref?t={int(time.time())}">' if reference_path() else '<div class="hint">No reference image yet.</div>')
    ck = ("Cookies loaded." if cookies_path() else "No cookies (not needed while Gemini watches and Vid Kraken downloads).")
    return (store_card()
            + '<form method="POST" action="/settings">' + form + '<div class="stick"><button class="bigbtn" type="submit">Save settings</button></div></form>'
            '<div class="card"><h2>Your animation style (reference image)</h2><div class="upl">' + ref +
            '<input type="file" accept="image/png,image/jpeg,image/webp" onchange="up(this)"><span id="upmsg" class="hint"></span></div>'
            '<div class="hint">Krea copies the look of this image. The free Cloudflare fallback can\'t use it.</div></div>'
            + krea_card() +
            '<div class="card"><h2>YouTube cookies (downloader fallback only)</h2><div class="hint" style="margin:0 0 8px">' + ck + '</div>'
            '<input type="file" accept=".txt" onchange="upc(this)"> <span id="upc" class="hint"></span></div>')


def store_card():
    s = storage_status()
    if s:
        return ('<div class="card ok">💾 Settings, keys, style image and Krea login survive redeploys: ' + esc(", ".join(s)) + ".</div>")
    return ('<div class="card warn">⚠️ Everything saved here (keys too) is lost on every redeploy. Fix: Railway → your service → Settings → '
            'Volumes → add a volume mounted at <code>/data</code> (also keeps the video + image banks). '
            'Or set <code>BACKUP_REPO</code> (a PRIVATE GitHub repo) + <code>GITHUB_TOKEN</code> (settings and keys only, not the banks).</div>')


def bank_html():
    rows = ""
    cur = job_name(last_job_dir())
    try:
        dirs = sorted(os.listdir(JOBS_DIR), key=lambda d: os.path.getmtime(os.path.join(JOBS_DIR, d)), reverse=True)
    except OSError:
        dirs = []
    for d in dirs:
        jd = os.path.join(JOBS_DIR, d)
        if not os.path.isdir(jd) or d.endswith("_src"):
            continue
        vd = jread(jd, "video.json") or {}
        part = vd.get("part") or [1, 1]
        tp = os.path.join(jd, "thumbnail.jpg")
        img = f'<img src="/jf/{esc(d)}/thumbnail.jpg?v={_mt(tp)}">' if os.path.isfile(tp) else '<div class="nt"></div>'
        badges = ""
        if d == cur:
            badges += '<span class="chip ok">viewing</span>'
        if part[1] > 1:
            badges += f'<span class="chip">part {part[0]}/{part[1]}</span>'
        if jread(jd, "posted.json"):
            badges += '<span class="chip ok">posted</span>'
        elif jread(jd, "done.json") and review_reasons(jd) and not is_approved(jd):
            badges += '<span class="chip bad">needs approval</span>'
        rows += (f'<div class="job">{img}<div><h4>{esc((vd.get("title") or d)[:80])}</h4>{badges}<span class="m">{dir_mb(jd):.0f} MB</span><br>'
                 f'<form method="POST" action="/bank/open" onsubmit="try{{localStorage.setItem(\'tab\',\'results\')}}catch(e){{}}">'
                 f'<input type="hidden" name="id" value="{esc(d)}"><button class="ghost small" type="submit">Open</button></form> '
                 f'<form method="POST" action="/bank/delete" onsubmit="return confirm(\'Delete this saved video?\')">'
                 f'<input type="hidden" name="id" value="{esc(d)}"><button class="danger small" type="submit">Delete</button></form></div></div>')
    vids = ""
    try:
        for n in sorted(os.listdir(VBANK_DIR)):
            if n.endswith((".json", ".part")):
                continue
            vid_id = n.rsplit(".", 1)[0]
            vids += (f'<div class="krow"><code>{esc(n)}</code><span class="m">{os.path.getsize(os.path.join(VBANK_DIR, n)) / 1_000_000:.0f} MB</span>'
                     f'<a class="btn ghost small" style="padding:6px 11px" href="/dl/source/{esc(vid_id)}">⬇ Download</a></div>')
    except OSError:
        pass
    note = "" if volume_on() else '<div class="alert warn" style="margin:0 0 10px">No Railway volume: the banks are wiped on every redeploy. Add a volume mounted at /data.</div>'
    return ('<div class="card"><h2>Saved videos (job bank)</h2>' + note + '<div class="jobs">' + (rows or '<div class="empty">Nothing saved yet.</div>') + '</div></div>'
            '<div class="card"><h2>Source video bank</h2><div class="hint" style="margin:0 0 6px">Kept so Vid Kraken is only used once per video.</div>'
            + (vids or '<div class="hint">Empty.</div>') + '</div>'
            '<div class="card"><h2>Clear</h2><form method="POST" action="/bank/clear" onsubmit="return confirm(\'Delete these?\')">'
            '<select name="what"><option value="jobs">All saved videos + images</option><option value="videos">Source video bank</option>'
            '<option value="all">Everything it made</option></select><button class="danger" style="margin-top:10px" type="submit">Clear</button></form></div>')


def render_page():
    rep = {"@@PILL@@": pill_html(), "@@KILL@@": killbox_html(), "@@TICKER@@": ticker_html(), "@@ALERT@@": alert_html(),
           "@@BUILD@@": build_html(), "@@RESULTS@@": results_html(), "@@KEYS@@": keys_html(),
           "@@SETTINGS@@": settings_html(), "@@BANK@@": bank_html(), "@@RUNNING@@": "true" if busy() else "false"}
    page = PAGE
    for k, v in rep.items():
        page = page.replace(k, v)
    return page


def partial_json():
    return json.dumps({"run": busy(), "pill": pill_html(), "kill": killbox_html(), "ticker": ticker_html(), "alert": alert_html(),
                       "progress": progress_html(), "activity": activity_html(), "results": results_html()})


def clear_banks(what):
    if what in ("jobs", "all"):
        shutil.rmtree(JOBS_DIR, ignore_errors=True)
        os.makedirs(JOBS_DIR, exist_ok=True)
        try:
            os.remove(LAST_JOB)
        except OSError:
            pass
    if what in ("videos", "all"):
        shutil.rmtree(VBANK_DIR, ignore_errors=True)
        os.makedirs(VBANK_DIR, exist_ok=True)
    if what == "all":
        for n in os.listdir(OUT_DIR):
            try:
                os.remove(os.path.join(OUT_DIR, n))
            except OSError:
                pass
    status(f"🗑️ Cleared: {esc(what)}.")


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

    def _redir(self, loc):
        self.send_response(303)
        self.send_header("Location", loc)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _form(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode() if length else ""
        return urllib.parse.parse_qs(raw, keep_blank_values=True)

    def _serve_file(self, path, download=None):
        """Serves a file with Range support (so videos seek and play on phones) and optional 'save as'."""
        if not path or not os.path.isfile(path):
            return self._send(404, b"not found")
        size = os.path.getsize(path)
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            s, _, e = rng[6:].partition("-")
            try:
                start = int(s) if s else max(0, size - int(e))
                end = min(int(e) if (e and s) else size - 1, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                code = 206
            except ValueError:
                start, end, code = 0, size - 1, 200
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".mp4"):
            ctype = "video/mp4"
        if path.endswith(".txt"):
            ctype = "text/plain; charset=utf-8"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{_slug(os.path.splitext(download)[0])}{os.path.splitext(download)[1]}"')
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-cache")
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
        if path == "/krea/connect":
            try:
                host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host", "")
                proto = (self.headers.get("X-Forwarded-Proto") or ("http" if host.startswith(("localhost", "127.")) else "https")).split(",")[0].strip()
                return self._redir(oauth_begin(f"{proto}://{host}"))
            except Exception as e:
                return self._send(500, f"Couldn't start the Krea sign-in: {e}".encode())
        if path == "/krea/callback":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                if q.get("error"):
                    raise RuntimeError((q.get("error_description") or q["error"])[0])
                oauth_finish((q.get("code") or [""])[0], (q.get("state") or [""])[0])
                status("🔑 Krea connected. Open Settings and tap 'Test the model ladder' to check your credits work.")
                return self._redir("/")
            except Exception as e:
                status(f"❌ Krea sign-in failed: {esc(str(e)[:300])}")
                return self._send(500, f"Krea sign-in failed: {e}".encode())
        if path == "/":
            if "partial=1" in urllib.parse.urlparse(self.path).query:
                return self._send(200, partial_json().encode(), "application/json; charset=utf-8")
            return self._send(200, render_page().encode(), "text/html; charset=utf-8")
        m = re.match(r"^/(dl|jf)/([\w-]+)/(final\.mp4|thumbnail\.jpg|script\.txt)$", path)
        if m:
            jd = os.path.join(JOBS_DIR, m.group(2))
            vd = jread(jd, "video.json") or {}
            meta = jread(jd, "meta.json") or {}
            nm = _slug(meta.get("title") or vd.get("title")) + {"final.mp4": ".mp4", "thumbnail.jpg": "-thumbnail.jpg", "script.txt": "-script.txt"}[m.group(3)]
            return self._serve_file(os.path.join(jd, m.group(3)), nm if m.group(1) == "dl" else None)
        m = re.match(r"^/dl/source/([\w-]+)$", path)
        if m:
            p, _meta = bank_find(m.group(1))
            return self._serve_file(p, os.path.basename(p) if p else None)
        if path.startswith("/job/"):
            jd = last_job_dir()
            rel = urllib.parse.unquote(path[5:])
            if jd and re.match(r"^((audio|images|frames|thumb)/[\w.-]+|thumbnail\.jpg|final\.mp4)$", rel):
                return self._serve_file(os.path.join(jd, rel))
            return self._send(404, b"not found")
        if path == "/out/preview.mp4":
            jd = last_job_dir()
            return self._serve_file(os.path.join(jd, "final.mp4") if jd else "")
        if path == "/out/script.txt":
            jd = last_job_dir()
            return self._serve_file(os.path.join(jd, "script.txt") if jd else "")
        if re.match(r"^/out/krea_test[\w]*\.png$", path):
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
            backup_file(COOKIES_SAVED)
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
            if e != ext:
                backup_delete("reference" + e)
        newp = os.path.join(DATA_DIR, "reference" + ext)
        with open(newp, "wb") as f:
            f.write(data)
        backup_file(newp)
        status("🖼️ Style reference image updated.")
        self._send(200, b"ok")

    def do_POST(self):
        if not self._authorized():
            return self._deny()
        ppath = urllib.parse.urlparse(self.path).path
        if ppath == "/kill":                        # always allowed, never blocked by anything
            kill_everything()
            return self._redir("/")
        if ppath == "/resume":
            resume_bot()
            return self._redir("/")
        if ppath == "/krea/test":
            if PAUSED:
                status("⛔ The bot is paused - resume first.")
            else:
                threading.Thread(target=krea_test, daemon=True).start()
            return self._redir("/")
        if ppath == "/krea/disconnect":
            try:
                os.remove(KREA_OAUTH_FILE)
            except OSError:
                pass
            backup_delete("krea_oauth.json")
            with _mcp["lock"]:
                _mcp["tools"] = None
            status("🔌 Krea disconnected.")
            return self._redir("/")
        if ppath == "/settings":
            try:
                settings_from_form(self._form())
                status("💾 Settings saved.")
            except Exception as e:
                status(f"❌ Couldn't save settings: {esc(str(e)[:200])}")
            return self._redir("/")
        if ppath == "/keys/add":
            f = self._form()
            name = (f.get("name") or [""])[0]
            try:
                keys_add(name, (f.get("key") or [""])[0])
                status(f"🔑 New {esc(name)} key saved. It is tried first from now on.")
            except Exception as e:
                status(f"❌ {esc(str(e)[:200])}")
            return self._redir("/")
        if ppath == "/keys/remove":
            f = self._form()
            name = (f.get("name") or [""])[0]
            if name in KEY_NAMES:
                keys_remove(name, (f.get("kid") or [""])[0])
                status(f"🗑️ Removed a {esc(name)} key.")
            return self._redir("/")
        if ppath == "/keys/retry":
            name = (self._form().get("name") or [""])[0]
            key_reset(name if name in KEY_NAMES else None)
            status(f"🔁 Failed {esc(name)} keys will be tried again.")
            return self._redir("/")
        if ppath == "/viewed":
            d = re.sub(r"[^\w-]", "", (self._form().get("id") or [""])[0]) or job_name(last_job_dir())
            jd = os.path.join(JOBS_DIR, d)
            if d and os.path.isdir(jd):
                jwrite(jd, "viewed.json", {"at": int(time.time())})
            return self._send(200, b"ok")
        if ppath == "/approve":
            d = re.sub(r"[^\w-]", "", (self._form().get("id") or [""])[0]) or job_name(last_job_dir())
            jd = os.path.join(JOBS_DIR, d)
            if d and os.path.isdir(jd) and is_viewed(jd):
                jwrite(jd, "approved.json", {"at": int(time.time())})
                status("✅ Approved. You can post it now.")
            else:
                status("⚠️ Watch the video first (at least 80%), then approve it.")
            return self._redir("/")
        if ppath == "/post":
            d = re.sub(r"[^\w-]", "", (self._form().get("id") or [""])[0])
            if d and os.path.isdir(os.path.join(JOBS_DIR, d)) and not busy():
                with open(LAST_JOB, "w") as fh:
                    fh.write(d)
            threading.Thread(target=run_post, daemon=True).start()
            return self._redir("/")
        if ppath == "/bank/open":
            d = re.sub(r"[^\w-]", "", (self._form().get("id") or [""])[0])
            if d and os.path.isdir(os.path.join(JOBS_DIR, d)) and not busy():
                with open(LAST_JOB, "w") as fh:
                    fh.write(d)
                status(f"📂 Opened the saved video {esc(d)}.")
            elif busy():
                status("⏳ Wait for the running job to finish before switching.")
            return self._redir("/")
        if ppath == "/bank/delete":
            d = re.sub(r"[^\w-]", "", (self._form().get("id") or [""])[0])
            if busy():
                status("⏳ Stop or wait for the running job before deleting.")
            elif d and os.path.isdir(os.path.join(JOBS_DIR, d)):
                shutil.rmtree(os.path.join(JOBS_DIR, d), ignore_errors=True)
                status(f"🗑️ Deleted the saved video {esc(d)}.")
            return self._redir("/")
        if ppath == "/bank/clear":
            what = (self._form().get("what") or ["jobs"])[0]
            if busy():
                status("⏳ Stop or wait for the running job before clearing.")
            elif what in ("jobs", "videos", "all"):
                clear_banks(what)
            return self._redir("/")
        if ppath != "/run":
            return self._send(404, b"not found")
        f = self._form()
        handle = (f.get("handle") or [""])[0].strip()
        mode_v = (f.get("mode") or ["channel"])[0]
        try:
            seconds = int((f.get("seconds") or [str(PREVIEW_SECONDS)])[0])
        except ValueError:
            seconds = PREVIEW_SECONDS
        fresh = bool((f.get("fresh") or [""])[0])
        whole = (f.get("scope") or ["whole"])[0] == "whole"
        autopost = bool((f.get("autopost") or [""])[0])
        split = bool((f.get("split") or [""])[0])
        vis = (f.get("visibility") or ["private"])[0]
        if PAUSED:
            status("⛔ The bot is PAUSED (kill switch). Press Resume first.")
        elif handle and mode_v == "video" and not VIDEO_LINK_RE.search(handle):
            status("❌ That doesn't look like a link to ONE video (youtube.com/watch?v=..., youtu.be/..., /shorts/...). "
                   "Switch to Channel mode if it's a channel.")
        elif handle:
            try:
                settings_update({"LAST_HANDLE": handle, "LAST_MODE": mode_v, "PREVIEW_SECONDS": str(seconds),
                                 "LAST_WHOLE": "1" if whole else "0", "AUTO_POST": "1" if autopost else "0",
                                 "SPLIT_IN_TWO": "1" if split else "0", "POST_VISIBILITY": vis})
            except Exception as e:
                print(f"couldn't save run settings: {e}", flush=True)
            status(f"▶️ Build started for {esc(handle)}" + (" (whole video)" if whole else f" (first {seconds}s)")
                   + (", split in two" if split else "") + (", fresh start" if fresh else "")
                   + (", will auto-post." if autopost else "."))
            threading.Thread(target=run_job, args=(handle, seconds, fresh, whole), daemon=True).start()
        self._redir("/")


def main():
    if len(sys.argv) > 1:                       # CLI mode
        load_cookies_env()
        init_storage()
        startup_update()
        out = build_preview(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
        print("Done:", ", ".join(out))
        return
    port = int(os.environ.get("PORT", 8080))
    load_cookies_env()
    init_storage()
    threading.Thread(target=startup_update, daemon=True).start()
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever, daemon=True).start()
    print(f"Faceless bot dashboard on port {port}", flush=True)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
