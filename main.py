"""Faceless YouTube bot - PREVIEW build (no posting yet).

What it does, step by step:
  1. You give it a YouTube channel handle (e.g. @somechannel).
  2. It finds the channel's latest long-form video.
  3. Gemini WATCHES the video (opens the public YouTube link itself) and returns the
     scenes: what is said + what is on screen. Fallback: yt-dlp subtitles + scene detection.
  4. The AI rewrites the narration of every scene: new sentence structure,
     same meaning (facts, numbers, order of ideas all kept).
  5. Voice-over is generated for each scene (Fish Audio -> ElevenLabs -> Microsoft free voice;
     Gemini voice only if enabled in Settings).
  6. Krea redraws each scene in YOUR animation style (your reference image).
     If the next scene looks the same as the previous one, the image is reused.
  7. Every image gets a slow zoom / pan that lasts exactly as long as that
     scene's voice-over, then everything is stitched into one video.

Output: data/out/preview.mp4 + data/out/script.txt (original vs rewritten, per
scene, so you can check the meaning really stayed the same).

PERSISTENCE (so nothing you enter is lost on a Railway redeploy)
  Everything you enter on the dashboard (Settings card, last channel, style image, cookies,
  Krea login) is saved to the data folder. Railway wipes that folder on every deploy unless
  you do ONE of these:
    A) Add a Railway Volume (service -> Settings -> Volumes -> mount path /data).
       Railway sets RAILWAY_VOLUME_MOUNT_PATH automatically and the bot uses it. Done.
    B) No volume: set BACKUP_REPO=youruser/yourprivaterepo and GITHUB_TOKEN (a token with
       "Contents: read & write" on that repo). The bot copies your saved files into that repo
       and restores them on start. The repo MUST be private (it holds your keys).
  You can use both. The local copy wins when it exists.
  BACKUP_BRANCH   optional, default main
  BACKUP_TOKEN    optional, use instead of GITHUB_TOKEN
  DATA_DIR        optional, overrides the data folder

Env vars (all of these can ALSO be typed into the dashboard Settings card, which wins over env):
  GEMINI_API_KEY      Google Gemini key (FREE tier works) - watch + rewrite + frame description
                      get one at aistudio.google.com -> Get API key
  ANTHROPIC_API_KEY   optional alternative to Gemini (paid). Gemini is used if both are set
  GEMINI_MODEL        optional, default gemini-2.5-flash (if Google removed it, the bot
                      automatically picks the newest Gemini Flash model)
  WATCH_WITH_GEMINI   optional, set to 0 to skip the Gemini 'watch' step
  KREA_API_KEY        Krea API token (krea.ai/settings/api-tokens)
  FISH_API_KEY        Fish Audio key (optional - skipped automatically if it has no credit)
  FISH_VOICE_ID       optional, your Fish voice model id (default voice if empty)
  FISH_MODEL          optional, default s2-pro
  ELEVENLABS_API_KEY  ElevenLabs key (ELEVEN_VOICE_ID / ELEVEN_MODEL optional)
  Voice order: Fish Audio -> ElevenLabs -> (Gemini voice, if enabled) -> free Microsoft voice.
  If a voice fails part-way, ALL lines are redone with the next voice so the voice never changes.
  Images: Krea via your web credits (Connect Krea on the dashboard), then Krea API key, then the free worker.
  CF_IMAGE_URL        your free Cloudflare Worker URL (CF_IMAGE_KEY defaults to 12345678)
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
  DASHBOARD_PASSWORD  optional (strongly recommended: the dashboard holds your keys)
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

# ---------------------------------------------------------------------------
# Settings (defaults come from env vars; the dashboard Settings card overrides them)
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
USE_GEMINI_VOICE = os.environ.get("USE_GEMINI_VOICE", "0").lower() in ("1", "true", "yes")   # tiny free daily quota
ELEVEN_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "") or os.environ.get("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "pNInz6obpgDQGcFmaJgB")   # "Adam" (free-plan premade voice)
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
os.makedirs(OUT_DIR, exist_ok=True)
REF_EXT = (".png", ".jpg", ".jpeg", ".webp")

# Fields editable on the dashboard (saved in data/settings.json). (name, label, kind)
SETTING_FIELDS = [
    ("GEMINI_API_KEY", "Gemini API key", "secret"),
    ("ANTHROPIC_API_KEY", "Anthropic API key (optional)", "secret"),
    ("KREA_API_KEY", "Krea API key (optional)", "secret"),
    ("FISH_API_KEY", "Fish Audio key (optional)", "secret"),
    ("FISH_VOICE_ID", "Fish voice id (optional)", "text"),
    ("ELEVEN_API_KEY", "ElevenLabs key (optional)", "secret"),
    ("ELEVEN_VOICE_ID", "ElevenLabs voice id", "text"),
    ("CF_IMAGE_URL", "Cloudflare image worker URL", "text"),
    ("CF_IMAGE_KEY", "Cloudflare worker key", "secret"),
    ("EDGE_VOICE", "Free Microsoft voice", "text"),
    ("STYLE_HINT", "Style hint added to every image prompt", "text"),
    ("STYLE_STRENGTH", "Style strength (0 to 1)", "float"),
    ("PREVIEW_SECONDS", "Default seconds to rebuild", "int"),
    ("USE_GEMINI_VOICE", "Try the Gemini voice before the free Microsoft voice", "bool"),
]
ENV_DEFAULTS = {k: globals()[k] for k, _, _ in SETTING_FIELDS}
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

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
            except Exception as e:
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
            except Exception as e:
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


def init_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    restore_from_github()
    apply_settings()
    s = storage_status()
    status("💾 Saved data survives redeploys via: " + ", ".join(s) if s else
           "⚠️ Nothing keeps your saved data across redeploys yet - add a Railway volume or set BACKUP_REPO.")


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
# Krea: your style -> images (API key route)
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
# Voice-over: Fish Audio -> ElevenLabs -> (Gemini, optional) -> free Microsoft voice
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
    missing = [] if (KREA_API_KEY or CF_IMAGE_URL or krea_connected()) else ["KREA_API_KEY, CF_IMAGE_URL or a Krea connection"]
    if not (GEMINI_API_KEY or ANTHROPIC_API_KEY):
        missing.insert(0, "GEMINI_API_KEY (free at aistudio.google.com)")
    if missing:
        raise RuntimeError("missing settings: " + ", ".join(missing))
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is not installed")
    if not reference_path():
        status("ℹ️ No style reference image uploaded - images will use Krea's default look.")


# ---------------------------------------------------------------------------
# Krea through its MCP server + OAuth: spends your normal Krea web credits (not the separate API balance)
# ---------------------------------------------------------------------------
KREA_MCP_URL = "https://api.krea.ai/mcp"
KREA_OAUTH_FILE = os.path.join(DATA_DIR, "krea_oauth.json")
_oauth_lock = threading.RLock()
_oauth_pending = {}
_mcp = {"sid": None, "tools": None, "lock": threading.RLock(), "n": 0}
_mcp_variant = [0]


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
        tmp = os.path.join(tempfile.gettempdir(), "ref_small.jpg")
        run_ffmpeg(["ffmpeg", "-y", "-i", pth, "-vf", "scale='min(1024,iw)':-2", "-q:v", "4", tmp], "shrink reference")
        with open(tmp, "rb") as f:
            _ref_uri[key] = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    return _ref_uri[key]


def krea_mcp_image(prompt, ref_uri, dest):
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
    box = _pick_prop(gen, ["input", "inputs", "params", "parameters", "arguments", "body"])
    model = KREA_MODEL if KREA_MODEL.startswith(("image/", "video/")) else "image/" + KREA_MODEL
    base = {"prompt": prompt}
    if ref_uri:
        base["image_style_references"] = [{"url": ref_uri, "strength": STYLE_STRENGTH}]
    jobtool = _find_tool(tools, ["get_job", "job_status", "check_job", "job"])
    err = ""
    for vi in range(_mcp_variant[0], len(SIZE_VARIANTS)):
        inputs = {**base, **SIZE_VARIANTS[vi]}
        args = {}
        if model_key in props:
            args[model_key] = model
        if box:
            args[box] = inputs
        else:
            args.update(inputs)
        try:
            data = mcp_tool(gname, args)
        except RuntimeError as e:
            msg = str(e)
            if any(w in msg.lower() for w in ("compute", "credit", "insufficient", "billing", "balance", "quota", "402")):
                raise
            err = msg
            continue
        _mcp_variant[0] = vi
        jid = _find_key(data, ["job_id", "jobId", "id"])
        t0 = time.time()
        while True:
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
        return
    raise RuntimeError(f"Krea MCP rejected the request: {err[:300]}")


def krea_test():
    try:
        status("🧪 Testing Krea with your credits ...")
        with _mcp["lock"]:
            _mcp["tools"] = None
        mcp_ready()
        tools = _mcp["tools"]
        status("🧪 Connected. Krea tools: " + esc(", ".join(tools) or "none"))
        for n, t in tools.items():
            print(f"KREA TOOL {n}: {json.dumps((t.get('inputSchema') or {}).get('properties') or {})[:500]}", flush=True)
        gname = _find_tool(tools, ["generate_image", "image_generate", "generate", "create_image", "text_to_image", "image"])
        if gname:
            g = tools[gname]
            status(f"🧪 Using tool <b>{esc(gname)}</b>, inputs: "
                   + esc(json.dumps((g.get("inputSchema") or {}).get("properties") or {})[:600]))
        else:
            status("🧪 No image-generation tool found in that list (full schemas are in the Railway logs).")
        dest = os.path.join(OUT_DIR, "krea_test.png")
        krea_mcp_image("a friendly cartoon caveman sitting by a campfire at night", reference_data_uri(), dest)
        status('✅ Krea works with your credits: <a href="/out/krea_test.png" target="_blank">see the test image</a>')
    except Exception as e:
        status(f"❌ Krea test failed: {esc(str(e)[:600])}")


def krea_card():
    if krea_connected():
        extra = ""
        if not storage_status():
            extra = ('<details style="margin-top:12px"><summary>Keep it after redeploys (no storage set up)</summary><div class="sb">'
                     'Best fix: add a Railway volume or set BACKUP_REPO. Otherwise paste this into the '
                     '<code>KREA_OAUTH_BLOB</code> env var. It is a secret login, keep it private.'
                     f'<textarea readonly rows="4" style="width:100%;margin-top:8px;background:#08080d;color:#eaeaf2;border:1px solid #26263a;border-radius:8px">{esc(krea_blob())}</textarea>'
                     '</div></details>')
        return ('<div class="card"><h2>Krea (your web credits)</h2><div class="hint" style="margin:0 0 8px">'
                '&#9989; Connected. Images use your normal Krea compute credits first.</div>'
                '<form method="POST" action="/krea/test" style="display:inline"><button type="submit">Test it</button></form> '
                '<form method="POST" action="/krea/disconnect" style="display:inline"><button type="submit" style="background:#26263a">Disconnect</button></form>'
                + extra + '</div>')
    return ('<div class="card"><h2>Krea (your web credits)</h2><div class="hint" style="margin:0 0 8px">'
            'Not connected. Sign in once so the bot can spend your normal Krea credits. On the Krea screen, '
            'pick the workspace that has the credits.</div>'
            '<a href="/krea/connect"><button type="button">Connect Krea</button></a></div>')


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


def make_images(unique, prompts, imgdir):
    """Try the image providers in order. The first that draws EVERY image wins; if one fails on the
    first image or runs out part-way, all images are redone with the next one (consistent style)."""
    options = []
    if krea_connected():
        options.append(("Krea (your credits)", lambda pr, d: krea_mcp_image(
            pr + (". " + STYLE_HINT if STYLE_HINT else ""), reference_data_uri(), d), WORKERS))
    if KREA_API_KEY:
        options.append(("Krea API", lambda pr, d: krea_image(pr + (". " + STYLE_HINT if STYLE_HINT else ""),
                                                          krea_reference_url(), d), WORKERS))
    if CF_IMAGE_URL:
        options.append(("Cloudflare AI (free)", lambda pr, d: cf_image(pr + ". " + (STYLE_HINT or CF_STYLE), d), 2))
    if not options:
        raise RuntimeError("no image provider: connect Krea, set a Krea API key or a Cloudflare worker URL in Settings")

    def dest(i):
        return os.path.join(imgdir, f"img{i:03d}.png")

    for k, (name, fn, workers) in enumerate(options):
        last = k == len(options) - 1
        try:
            fn(prompts[unique[0]], dest(unique[0]))
        except Exception as e:
            if last:
                raise
            why = "no API balance" if "402" in str(e) else str(e)[:400]
            print(f"{name} failed: {e}", flush=True)
            status(f"⚠️ {name} unavailable ({esc(why)}). Trying the next image provider ...")
            continue
        status(f"🎨 Images: {name}")
        try:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(lambda i: fn(prompts[i], dest(i)), unique[1:]))
        except Exception as e:
            if last:
                raise
            status(f"⚠️ {name} failed mid-way ({esc(str(e)[:200])}). Redoing all images with the next provider ...")
            continue
        return name


# --- saved progress per video (lets a failed run resume, and feeds the dashboard cards) ---
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


def reset_job(jd, vid):
    shutil.rmtree(jd, ignore_errors=True)
    os.makedirs(os.path.join(jd, "audio"), exist_ok=True)
    os.makedirs(os.path.join(jd, "images"), exist_ok=True)
    jwrite(jd, "video.json", {"title": vid["title"], "url": vid["url"]})


def build_preview(handle, seconds=None, fresh=False):
    check_setup()
    seconds = max(20, int(seconds or PREVIEW_SECONDS))
    work = tempfile.mkdtemp(prefix="fl_")
    job_error[0] = ""
    try:
        status(f"🔎 Looking up the latest video on {esc(handle)} ...")
        vid = yt_call(latest_video, handle)
        status(f"📺 Latest video: {esc(vid['title'][:90])}")

        vkey = re.sub(r"[^\w-]", "", vid["id"]) or "video"
        jd = os.path.join(JOBS_DIR, vkey)
        for d in os.listdir(JOBS_DIR):                       # keep only this video's saved progress
            if d != vkey:
                shutil.rmtree(os.path.join(JOBS_DIR, d), ignore_errors=True)
        sc = None if fresh else jread(jd, "scenes.json")
        if sc and sc.get("seconds") == seconds and sc.get("scenes") and sc.get("owner"):
            jwrite(jd, "video.json", {"title": vid["title"], "url": vid["url"]})
        else:
            sc = None
            reset_job(jd, vid)
        try:
            os.remove(os.path.join(jd, "done.json"))
        except OSError:
            pass
        with open(LAST_JOB, "w") as f:
            f.write(vkey)

        # ---- stage 2: scenes ----
        if sc:
            scenes, visuals, owner = sc["scenes"], sc.get("visuals"), sc["owner"]
            n = len(scenes)
            frames = [None] * n
            unique = sorted(set(owner))
            status(f"♻️ Reusing the {n} scenes saved from the last run (tick 'Start fresh' to redo everything).")
        else:
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
            jwrite(jd, "scenes.json", {"seconds": seconds, "scenes": scenes, "visuals": visuals, "owner": owner})

        # ---- stage 3: rewrite ----
        rw = jread(jd, "rewrite.json")
        if rw and len(rw.get("text") or []) == n:
            new_text = rw["text"]
            status("♻️ Reusing the saved rewrite.")
        else:
            status("✍️ Rewriting the narration (same meaning, new structure) ...")
            new_text = rewrite_scenes(scenes, vid["title"])
            for stale in ("voice.json", "images.json"):
                try:
                    os.remove(os.path.join(jd, stale))
                except OSError:
                    pass
            jwrite(jd, "rewrite.json", {"text": new_text})

        # ---- stage 4: voice-over ----
        audios = [os.path.join(jd, "audio", f"a{i:03d}.mp3") for i in range(n)]
        vo = jread(jd, "voice.json")
        if vo and all(os.path.isfile(a) and os.path.getsize(a) > 500 for a in audios):
            voice_used = vo.get("name", "")
            status(f"♻️ Reusing the saved voice-over ({esc(voice_used)}).")
        else:
            status("🎙️ Generating voice-over ...")
            voice_used = make_voiceovers(new_text, audios)
            jwrite(jd, "voice.json", {"name": voice_used})

        # ---- stage 5: images ----
        imgdir = os.path.join(jd, "images")
        images = {i: os.path.join(imgdir, f"img{i:03d}.png") for i in unique}
        im = jread(jd, "images.json")
        if im and all(os.path.isfile(pth) for pth in images.values()):
            prompts = {int(k): v for k, v in (im.get("prompts") or {}).items()}
            provider = im.get("provider", "")
            status(f"♻️ Reusing the saved images ({esc(provider)}).")
        else:
            for fn_ in os.listdir(imgdir):
                try:
                    os.remove(os.path.join(imgdir, fn_))
                except OSError:
                    pass
            status(f"🎨 Redrawing {len(unique)} frames ...")
            prompts = {}

            def get_prompt(i):
                prompts[i] = visuals[i] if visuals else describe_frame(frames[i], scenes[i][2])

            with ThreadPoolExecutor(WORKERS) as ex:
                list(ex.map(get_prompt, unique))
            provider = make_images(unique, prompts, imgdir)
            jwrite(jd, "images.json", {"provider": provider, "prompts": {str(k): v for k, v in prompts.items()}})

        # ---- stage 6: final video ----
        status("🎞️ Animating scenes and joining them ...")
        parts = []
        for i in range(n):
            dst = os.path.join(work, f"s{i:03d}.mp4")
            scene_clip(images[owner[i]], audios[i], i, dst)
            parts.append(dst)
        final = os.path.join(work, "final.mp4")
        concat(parts, final, work)
        shutil.move(final, os.path.join(OUT_DIR, "preview.mp4"))

        lines = [f"SOURCE: {vid['title']}  ({vid['url']})", f"VOICE: {voice_used}", f"IMAGES: {provider}", ""]
        for i, (a, b, t) in enumerate(scenes):
            lines += [f"--- Scene {i + 1}  [{a:.1f}s-{b:.1f}s]  image from scene {owner[i] + 1}",
                      f"ORIGINAL : {t}", f"REWRITTEN: {new_text[i]}",
                      f"IMAGE PROMPT: {prompts.get(owner[i], '')}", ""]
        with open(os.path.join(OUT_DIR, "script.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        jwrite(jd, "done.json", {"voice": voice_used, "images": provider})

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


def run_job(handle, seconds, fresh=False):
    global job_running
    if not job_lock.acquire(blocking=False):
        status("⏳ A preview is already being built.")
        return
    job_running = True
    try:
        build_preview(handle, seconds, fresh)
    except Exception as e:
        print(f"Job failed: {type(e).__name__}: {e}", flush=True)
        job_error[0] = str(e)[:300] or "failed"
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
input[type=text],input[type=number],input[type=password]{background:var(--p2);border:1px solid var(--border);border-radius:8px;padding:9px 10px;color:var(--text);font:inherit;width:100%}
button{background:linear-gradient(135deg,var(--a),var(--a2));color:#fff;border:0;padding:11px 16px;border-radius:10px;font-size:14px;font-weight:600;cursor:pointer;margin-top:14px}
.feed{display:flex;flex-direction:column;gap:7px;max-height:360px;overflow-y:auto}
.fi{background:#08080d;border:1px solid var(--border);border-radius:10px;padding:9px 12px;font-size:13px;line-height:1.5}
a{color:var(--a2);text-decoration:none}.hint{font-size:12px;color:var(--muted);margin-top:8px;line-height:1.5}
.warn{border-color:rgba(255,59,92,.4);color:#ffb3c0;font-size:13px}.ok{border-color:rgba(60,200,120,.35);color:#b6f0cf;font-size:13px}
video{width:100%;border-radius:12px;background:#000}
img.ref{max-height:140px;border-radius:10px;border:1px solid var(--border);display:block;margin-bottom:8px}
details{background:var(--p2);border:1px solid var(--border);border-radius:12px;margin-bottom:8px}
summary{padding:12px 14px;cursor:pointer;font-size:14px;font-weight:600}
.sb{padding:4px 14px 14px;font-size:13px;line-height:1.5}
.sc{border-top:1px solid var(--border);padding:8px 0}.m{color:var(--muted);font-size:12px}
.rw{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px}.rw div{background:#08080d;border-radius:8px;padding:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.grid img{width:100%;border-radius:8px;display:block}audio{width:100%;height:34px;margin-top:4px}
@media(max-width:560px){.rw{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>Faceless Bot - preview</h1><div class="sub">Channel in, rebuilt video out. Nothing is posted.</div>
@@WARN@@
@@STORE@@
<div class="card"><h2>1. Your animation style</h2>@@REF@@
<input type="file" accept="image/png,image/jpeg,image/webp" onchange="up(this)"> <span id="upmsg" class="hint"></span>
<div class="hint">Krea copies the look of this image. (The free Cloudflare fallback can't use it.)</div></div>
<div class="card"><h2>YouTube cookies (only for the downloader fallback)</h2><div class="hint" style="margin:0 0 8px">@@COOKIES@@</div>
<input type="file" accept=".txt" onchange="upc(this)"> <span id="upc" class="hint"></span></div>
@@KREA@@
@@SETTINGS@@
<div class="card"><h2>2. Build a preview</h2>
<form method="POST" action="/run"><label>YouTube channel handle or link</label>
<input type="text" name="handle" placeholder="@channelname" value="@@HANDLE@@" required>
<label>Seconds of the video to rebuild</label><input type="number" name="seconds" min="20" max="600" value="@@SECS@@">
<label style="display:flex;gap:8px;align-items:center;color:var(--text)"><input type="checkbox" name="fresh" value="1"> Start fresh (ignore saved progress)</label>
<button type="submit" id="runbtn">@@BTN@@</button></form>
<div class="hint">If a run fails, running again on the same video continues from the saved stages instead of paying for them twice.</div></div>
<div id="live">@@LIVE@@</div>
</div><script>
async function up(i){const f=i.files[0];if(!f)return;document.getElementById('upmsg').textContent='Uploading...';
const r=await fetch('/upload-ref?name='+encodeURIComponent(f.name),{method:'PUT',body:f});
if(!r.ok){document.getElementById('upmsg').textContent='Failed: '+await r.text();return}location.reload()}
async function upc(i){const f=i.files[0];if(!f)return;const r=await fetch('/upload-cookies',{method:'PUT',body:f});
if(!r.ok){document.getElementById('upc').textContent='Failed: '+await r.text();return}location.reload()}
let last="";
async function poll(){try{const r=await fetch('/?partial=1');const run=r.headers.get('X-Running')==='1';const t=await r.text();
if(t!==last){const open=[...document.querySelectorAll('#live details[open]')].map(d=>d.dataset.id);
document.getElementById('live').innerHTML=t;last=t;
document.querySelectorAll('#live details').forEach(d=>{if(open.includes(d.dataset.id))d.open=true})}
document.getElementById('runbtn').textContent=run?'Building...':'Build preview';
if(run)setTimeout(poll,4000)}catch(e){setTimeout(poll,6000)}}
@@RELOAD@@
</script></body></html>"""


def _mt(path):
    try:
        return int(os.path.getmtime(path))
    except OSError:
        return 0


def stages_html():
    jd = last_job_dir()
    if not jd:
        return ""
    vd = jread(jd, "video.json") or {}
    sc = jread(jd, "scenes.json") or {}
    rw = jread(jd, "rewrite.json") or {}
    vo = jread(jd, "voice.json") or {}
    im = jread(jd, "images.json") or {}
    dn = jread(jd, "done.json") or {}
    scenes = sc.get("scenes") or []
    visuals = sc.get("visuals") or []
    owner = sc.get("owner") or []
    texts = rw.get("text") or []
    prompts = {int(k): v for k, v in (im.get("prompts") or {}).items()}
    flags = [bool(vd), bool(scenes), bool(texts), bool(vo), bool(im), bool(dn)]
    pend = next((i for i, f in enumerate(flags) if not f), None)

    def icon(i):
        if flags[i]:
            return "✅"
        if i == pend and job_error[0]:
            return "❌"
        if i == pend and job_running:
            return "⏳"
        return "⚪"

    b0 = (f'<div>{esc(vd.get("title"))}</div><div class="m"><a href="{esc(vd.get("url"))}" target="_blank">'
          f'{esc(vd.get("url"))}</a></div>') if vd else ""

    b1 = ""
    for i, (a, b, t) in enumerate(scenes):
        vis = visuals[i] if i < len(visuals) and visuals[i] else ""
        b1 += (f'<div class="sc"><b>Scene {i + 1}</b> <span class="m">{a:.1f}s-{b:.1f}s</span><div>{esc(t)}</div>'
               + (f'<div class="m">Image prompt: {esc(vis)}</div>' if vis else "") + "</div>")

    b2 = ""
    for i, (a, b, t) in enumerate(scenes):
        if i < len(texts):
            b2 += (f'<div class="sc"><b>Scene {i + 1}</b><div class="rw"><div><span class="m">Original</span><br>{esc(t)}</div>'
                   f'<div><span class="m">Rewritten</span><br>{esc(texts[i])}</div></div></div>')

    b3 = ""
    if vo:
        b3 = f'<div class="m">Voice used: {esc(vo.get("name"))}</div>'
        for i in range(len(texts)):
            ap = os.path.join(jd, "audio", f"a{i:03d}.mp3")
            if os.path.isfile(ap):
                b3 += (f'<div class="sc"><b>Scene {i + 1}</b><div>{esc(texts[i])}</div>'
                       f'<audio controls preload="none" src="/job/audio/a{i:03d}.mp3?v={_mt(ap)}"></audio></div>')

    b4 = ""
    if im:
        b4 = f'<div class="m">Drawn with: {esc(im.get("provider"))} &middot; tap an image to open it full size</div><div class="grid">'
        for i in sorted(set(owner)):
            ip = os.path.join(jd, "images", f"img{i:03d}.png")
            if os.path.isfile(ip):
                used = [str(k + 1) for k, o in enumerate(owner) if o == i]
                u = f"/job/images/img{i:03d}.png?v={_mt(ip)}"
                b4 += (f'<div><a href="{u}" target="_blank"><img loading="lazy" src="{u}"></a>'
                       f'<div class="m">Scene {esc(", ".join(used))}: {esc(prompts.get(i, ""))}</div></div>')
        b4 += "</div>"

    b5 = ""
    if dn:
        pv = os.path.join(OUT_DIR, "preview.mp4")
        b5 = (f'<video controls preload="metadata" src="/out/preview.mp4?t={_mt(pv)}"></video>'
              f'<div class="hint"><a href="/out/script.txt" target="_blank">Original vs rewritten script</a></div>')

    bodies = [b0, b1, b2, b3, b4, b5]
    if job_error[0] and pend is not None:
        bodies[pend] += f'<div class="m" style="color:#ffb3c0">Stopped here: {esc(job_error[0][:200])}</div>'

    n = len(scenes)
    vname = esc(vo.get("name")) if vo else ""
    iname = esc(im.get("provider")) if im else ""
    titles = ["1. Latest video found",
              "2. Gemini watched the video" + (f" ({n} scenes)" if n else ""),
              "3. Narration rewritten",
              "4. Voice-over" + (f" ({vname})" if vo else ""),
              "5. Images redrawn" + (f" ({iname})" if im else ""),
              "6. Final video"]
    return "".join(f'<details data-id="s{i}"><summary>{icon(i)} {titles[i]}</summary><div class="sb">{bodies[i]}</div></details>'
                   for i in range(6))


def live_html():
    with status_lock:
        items = list(status_feed[-30:])
    feed = "\n".join(f'<div class="fi">{i}</div>' for i in reversed(items)) or '<div class="hint">Nothing yet.</div>'
    st = stages_html()
    out = f'<div class="card"><h2>Stages (tap to open)</h2>{st}</div>' if st else ""
    return out + f'<div class="card"><h2>Activity</h2><div class="feed">{feed}</div></div>'


def store_card():
    s = storage_status()
    if s:
        return ('<div class="card ok">&#128190; Your settings, style image, cookies and Krea login survive redeploys: '
                + esc(", ".join(s)) + ".</div>")
    return ('<div class="card warn">&#9888; Saved data is lost on every redeploy. Fix (pick one): '
            '<b>1)</b> Railway: service &rarr; Settings &rarr; Volumes &rarr; add a volume mounted at <code>/data</code>. '
            '<b>2)</b> Set <code>BACKUP_REPO</code> (a PRIVATE GitHub repo, <code>user/repo</code>) and <code>GITHUB_TOKEN</code> '
            'in Railway variables.</div>')


def settings_card():
    rows = ""
    for key, label, kind in SETTING_FIELDS:
        cur = globals()[key]
        if kind == "bool":
            rows += (f'<label style="display:flex;gap:8px;align-items:center;color:var(--text)">'
                     f'<input type="checkbox" name="{key}" value="1"{" checked" if cur else ""}> {esc(label)}</label>')
        elif kind == "secret":
            ph = ("saved ····" + str(cur)[-4:]) if cur else "not set"
            rows += (f'<label>{esc(label)} <span class="m">(blank = keep, "-" = clear)</span></label>'
                     f'<input type="password" name="{key}" placeholder="{esc(ph)}" autocomplete="off">')
        else:
            rows += f'<label>{esc(label)}</label><input type="text" name="{key}" value="{esc(cur)}">'
    return ('<div class="card"><h2>Settings (saved)</h2><details><summary>Keys, voices and style</summary><div class="sb">'
            '<form method="POST" action="/settings">' + rows +
            '<button type="submit">Save settings</button></form>'
            '<div class="hint">Saved values win over Railway variables. They take effect immediately.</div></div></details></div>')


def render_page():
    warns = []
    if not (GEMINI_API_KEY or ANTHROPIC_API_KEY):
        warns.append("No Gemini key yet: add it under Settings below (free key at aistudio.google.com)")
    if not (KREA_API_KEY or CF_IMAGE_URL or krea_connected()):
        warns.append("No image provider: connect Krea below, or add a Krea API key / Cloudflare worker URL in Settings")
    warn = "".join(f'<div class="card warn">&#9888; {esc(w)}</div>' for w in warns)
    ref = f'<img class="ref" src="/ref?t={int(time.time())}">' if reference_path() else \
        '<div class="hint" style="margin:0 0 8px">No reference image yet.</div>'
    ck = ("Cookies loaded. If YouTube still blocks you, upload fresh ones." if cookies_path() else
          "No cookies. Not needed while Gemini can watch the video.")
    rep_ = {"@@STORE@@": store_card(), "@@SETTINGS@@": settings_card(), "@@KREA@@": krea_card(),
            "@@COOKIES@@": ck, "@@WARN@@": warn, "@@REF@@": ref, "@@LIVE@@": live_html(),
            "@@HANDLE@@": esc(settings_load().get("LAST_HANDLE", "")),
            "@@SECS@@": str(PREVIEW_SECONDS), "@@BTN@@": "Building..." if job_running else "Build preview",
            "@@RELOAD@@": "poll();" if job_running else ""}
    page = PAGE
    for k, v in rep_.items():
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

    def _redir(self, loc):
        self.send_response(303)
        self.send_header("Location", loc)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _form(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode() if length else ""
        return urllib.parse.parse_qs(raw, keep_blank_values=True)

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
                status("🔑 Krea connected. Tap 'Test it' to check your credits work.")
                return self._redir("/")
            except Exception as e:
                status(f"❌ Krea sign-in failed: {esc(str(e)[:300])}")
                return self._send(500, f"Krea sign-in failed: {e}".encode())
        if path == "/":
            if "partial=1" in urllib.parse.urlparse(self.path).query:
                body = live_html().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("X-Running", "1" if job_running else "0")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return self._send(200, render_page().encode(), "text/html; charset=utf-8")
        if path.startswith("/job/"):
            jd = last_job_dir()
            rel = urllib.parse.unquote(path[5:])
            if jd and re.match(r"^(audio|images)/[\w.-]+$", rel):
                return self._serve_file(os.path.join(jd, rel))
            return self._send(404, b"not found")
        if path in ("/out/preview.mp4", "/out/script.txt", "/out/krea_test.png"):
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
        if ppath == "/krea/test":
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
        if ppath != "/run":
            return self._send(404, b"not found")
        f = self._form()
        handle = (f.get("handle") or [""])[0].strip()
        try:
            seconds = int((f.get("seconds") or [str(PREVIEW_SECONDS)])[0])
        except ValueError:
            seconds = PREVIEW_SECONDS
        fresh = bool((f.get("fresh") or [""])[0])
        if handle:
            try:
                settings_update({"LAST_HANDLE": handle, "PREVIEW_SECONDS": str(seconds)})
            except Exception as e:
                print(f"couldn't save run settings: {e}", flush=True)
            status(f"▶️ Preview started for {esc(handle)}" + (" (fresh start)." if fresh else "."))
            threading.Thread(target=run_job, args=(handle, seconds, fresh), daemon=True).start()
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    if len(sys.argv) > 1:                       # CLI mode
        load_cookies_env()
        init_storage()
        startup_update()
        out = build_preview(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
        print("Done:", out)
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
