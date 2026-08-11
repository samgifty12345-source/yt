import os
import time
import json
import tempfile
import requests
import subprocess
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from huggingface_hub import InferenceClient

WORK_DIR = tempfile.gettempdir()
DONE_FILE = "done_history.txt"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Fish Audio TTS (https://docs.fish.audio) - primary narration voice.
FISH_AUDIO_API_KEY = os.environ.get("FISH_AUDIO_API_KEY", "")
FISH_AUDIO_VOICE_ID = os.environ.get("FISH_AUDIO_VOICE_ID", "")
FISH_AUDIO_MODEL = os.environ.get("FISH_AUDIO_MODEL", "s2.1-pro-free")

DEFAULT_NICHE = os.environ.get(
    "NICHE_PROMPT",
    "Ancient history and forgotten historical events, told dramatically but 100% factually accurate."
)

STYLE_SUFFIX = os.environ.get(
    "STYLE_SUFFIX",
    "flat 2D cartoon illustration style, muted earth tones, thick black outlines, consistent character design"
)

# How often autopilot posts once it's running on its own schedule.
AUTOPILOT_INTERVAL_HOURS = float(os.environ.get("AUTOPILOT_INTERVAL_HOURS", "24"))

# How long the bot waits after boot before its FIRST unattended run, so you
# have a window to open the site and set topic/duration/orientation first.
# Visiting the site and hitting "Generate Now" at any point cancels the wait.
STARTUP_WAIT_HOURS = float(os.environ.get("STARTUP_WAIT_HOURS", "2"))

HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")

HF_IMAGE_PROVIDERS = [
    p.strip() for p in os.environ.get("HF_IMAGE_PROVIDERS", "nscale,fal-ai,hf-inference").split(",")
    if p.strip()
]

POLLINATIONS_TOKEN = os.environ.get("POLLINATIONS_TOKEN", "")

# Point this at a direct URL to an mp3 you actually own/have rights to use
# (e.g. a file hosted in your own GitHub repo - NOT a signed/expiring link
# from a site like audio.com, and NOT a track marked "All Rights Reserved").
BACKGROUND_MUSIC_URL = os.environ.get("BACKGROUND_MUSIC_URL", "")
BACKGROUND_MUSIC_VOLUME = float(os.environ.get("BACKGROUND_MUSIC_VOLUME", "0.42"))

SCENE_SECONDS = 10  # each scene/image is on screen this long

# Orientation presets: (width, height, description used in prompts)
ORIENTATIONS = {
    "shorts": {"w": 1080, "h": 1920, "aspect_text": "vertical 9:16", "max_seconds": 180},
    "long":   {"w": 1920, "h": 1080, "aspect_text": "horizontal 16:9", "max_seconds": 600},
}

pipeline_log = ["History bot ready. Waiting for the startup window or a manual trigger."]
log_lock = threading.Lock()

pipeline_state_lock = threading.Lock()
pipeline_running = False

trigger_event = threading.Event()
next_run_at = [time.time() + STARTUP_WAIT_HOURS * 3600]  # mutable box for cross-thread read

config_lock = threading.Lock()
CONFIG = {
    "topic": DEFAULT_NICHE,
    "duration_seconds": 60,
    "orientation": "shorts",
}


def log(msg):
    print(msg, flush=True)
    with log_lock:
        pipeline_log.append(msg)
        if len(pipeline_log) > 80:
            pipeline_log.pop(0)


def get_config():
    with config_lock:
        return dict(CONFIG)


def get_google_creds(scopes):
    return Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        scopes=scopes,
    )


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI History Shorts Bot</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {
    --bg: #0b0b10;
    --panel: #14141c;
    --panel-2: #1b1b26;
    --border: #26263a;
    --text: #eaeaf2;
    --muted: #8a8aa0;
    --accent: #ff3b5c;
    --accent-2: #7c5cff;
    --ok: #35d488;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
    background:
      radial-gradient(1200px 600px at 10% -10%, rgba(124,92,255,0.18), transparent 60%),
      radial-gradient(1000px 500px at 100% 0%, rgba(255,59,92,0.14), transparent 55%),
      var(--bg);
    color: var(--text);
    padding: 32px 20px 60px;
  }
  .wrap { max-width: 880px; margin: 0 auto; }
  header { display: flex; align-items: center; gap: 14px; margin-bottom: 28px; }
  .logo {
    width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 18px; flex-shrink: 0;
  }
  h1 { font-size: 22px; margin: 0; letter-spacing: -0.02em; }
  .sub { color: var(--muted); font-size: 13px; margin-top: 2px; }
  .grid { display: grid; grid-template-columns: 1fr; gap: 18px; }
  @media (min-width: 720px) { .grid { grid-template-columns: 1fr 1fr; } }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
  }
  .card h2 {
    font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); margin: 0 0 16px;
  }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
  .badge {
    background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 999px; padding: 6px 12px; font-size: 12.5px; color: var(--muted);
  }
  .badge b { color: var(--text); }
  .badge.running { color: var(--ok); border-color: rgba(53,212,136,0.35); background: rgba(53,212,136,0.08); }
  label { display: block; font-size: 12.5px; color: var(--muted); margin: 14px 0 6px; }
  label:first-of-type { margin-top: 0; }
  textarea, input[type=number] {
    width: 100%; background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 12px; color: var(--text); font-size: 14px;
    font-family: inherit; resize: vertical;
  }
  textarea:focus, input:focus { outline: none; border-color: var(--accent-2); }
  .row { display: flex; gap: 10px; align-items: center; }
  .row input[type=number] { width: 90px; }
  .toggle { display: flex; gap: 8px; margin-top: 4px; }
  .toggle label {
    flex: 1; margin: 0; text-align: center; cursor: pointer;
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px; font-size: 13px; color: var(--muted); transition: all .15s;
  }
  .toggle input { display: none; }
  .toggle input:checked + label {
    color: #fff; border-color: var(--accent-2);
    background: linear-gradient(135deg, rgba(124,92,255,0.25), rgba(255,59,92,0.2));
  }
  button {
    width: 100%; margin-top: 18px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #fff; border: none; padding: 13px; border-radius: 10px;
    font-size: 14.5px; font-weight: 600; cursor: pointer; letter-spacing: 0.01em;
  }
  button:hover { filter: brightness(1.08); }
  button.secondary {
    background: var(--panel-2); border: 1px solid var(--border); color: var(--text);
  }
  .log {
    background: #08080d; border: 1px solid var(--border); border-radius: 10px;
    padding: 14px; font-size: 12px; color: #8fe3a8; font-family: "SF Mono", Menlo, Consolas, monospace;
    max-height: 360px; overflow-y: auto; white-space: pre-wrap; line-height: 1.5;
  }
  .hint { font-size: 11.5px; color: var(--muted); margin-top: 8px; line-height: 1.5; }
  footer { text-align: center; color: var(--muted); font-size: 11.5px; margin-top: 26px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">AI</div>
    <div>
      <h1>History Shorts Bot</h1>
      <div class="sub">Unattended AI research &rarr; script &rarr; voice &rarr; video &rarr; YouTube pipeline</div>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <h2>Status</h2>
      <div class="badges">
        <div class="badge">Posted <b>@@DONE_COUNT@@</b></div>
        <div class="badge">Autopilot every <b>@@INTERVAL@@h</b></div>
        <div class="badge @@RUNNING_CLASS@@">@@RUNNING_TEXT@@</div>
      </div>
      <div class="log">@@LOG_CONTENT@@</div>
      <form method="POST" action="/trigger">
        <button type="submit">Generate &amp; Post Now</button>
      </form>
      <div class="hint">Triggering now cancels any wait/schedule and starts immediately with the settings on the right.</div>
    </div>

    <div class="card">
      <h2>Configure Next Video</h2>
      <form method="POST" action="/configure">
        <label>Topic / niche</label>
        <textarea name="topic" rows="3" placeholder="e.g. The Bronze Age Collapse">@@TOPIC@@</textarea>

        <label>Orientation</label>
        <div class="toggle">
          <input type="radio" id="o_shorts" name="orientation" value="shorts" @@SHORTS_CHECKED@@>
          <label for="o_shorts">Short (vertical)</label>
          <input type="radio" id="o_long" name="orientation" value="long" @@LONG_CHECKED@@>
          <label for="o_long">Long-form (horizontal)</label>
        </div>

        <label>Duration</label>
        <div class="row">
          <input type="number" name="duration_minutes" min="0.25" max="10" step="0.25" value="@@DURATION_MINUTES@@">
          <span class="sub">minutes (~@@SCENE_COUNT@@ scenes)</span>
        </div>

        <button type="submit" class="secondary">Save Settings</button>
      </form>
      <div class="hint">
        Shorts are capped at 3 min, long-form at 10 min. Settings persist until you change them again,
        and apply to both manual and autopilot runs.
      </div>
    </div>
  </div>

  <footer>First unattended run waits @@STARTUP_WAIT@@h after boot &middot; next run in @@NEXT_RUN_IN@@</footer>
</div>
</body>
</html>"""


def format_countdown(target_epoch):
    remaining = int(target_epoch - time.time())
    if remaining <= 0:
        return "any moment"
    h, rem = divmod(remaining, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def render_page():
    done_count = 0
    if os.path.exists(DONE_FILE):
        with open(DONE_FILE) as f:
            done_count = len([l for l in f if l.strip()])
    with log_lock:
        log_text = "\n".join(pipeline_log[-40:])
    with pipeline_state_lock:
        running = pipeline_running

    cfg = get_config()
    duration_minutes = round(cfg["duration_seconds"] / 60, 2)
    scene_count = max(1, round(cfg["duration_seconds"] / SCENE_SECONDS))

    html = PAGE_TEMPLATE
    html = html.replace("@@DONE_COUNT@@", str(done_count))
    html = html.replace("@@INTERVAL@@", str(AUTOPILOT_INTERVAL_HOURS))
    html = html.replace("@@RUNNING_CLASS@@", "running" if running else "")
    html = html.replace("@@RUNNING_TEXT@@", "Running now" if running else "Idle")
    html = html.replace("@@LOG_CONTENT@@", log_text)
    html = html.replace("@@TOPIC@@", cfg["topic"])
    html = html.replace("@@SHORTS_CHECKED@@", "checked" if cfg["orientation"] == "shorts" else "")
    html = html.replace("@@LONG_CHECKED@@", "checked" if cfg["orientation"] == "long" else "")
    html = html.replace("@@DURATION_MINUTES@@", str(duration_minutes))
    html = html.replace("@@SCENE_COUNT@@", str(scene_count))
    html = html.replace("@@STARTUP_WAIT@@", str(STARTUP_WAIT_HOURS))
    html = html.replace("@@NEXT_RUN_IN@@", format_countdown(next_run_at[0]))
    return html


def parse_post_body(handler):
    length = int(handler.headers.get("Content-Length", 0))
    body = handler.rfile.read(length).decode() if length else ""
    return {k: v[0] for k, v in urllib.parse.parse_qs(body).items()}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = render_page()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        body = html.encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/configure":
            fields = parse_post_body(self)
            with config_lock:
                if fields.get("topic", "").strip():
                    CONFIG["topic"] = fields["topic"].strip()
                orientation = fields.get("orientation", CONFIG["orientation"])
                if orientation not in ORIENTATIONS:
                    orientation = CONFIG["orientation"]
                CONFIG["orientation"] = orientation
                try:
                    minutes = float(fields.get("duration_minutes", 1))
                except ValueError:
                    minutes = 1.0
                seconds = minutes * 60
                cap = ORIENTATIONS[orientation]["max_seconds"]
                seconds = max(SCENE_SECONDS, min(seconds, cap))
                CONFIG["duration_seconds"] = seconds
            log(f"Settings updated -> orientation={orientation}, "
                f"duration={seconds/60:.2f}min, topic='{CONFIG['topic'][:60]}'")
        elif self.path == "/trigger":
            log("Manual trigger received - starting now.")
            trigger_event.set()

        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


def start_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# ---------------------------------------------------------------------------
# Content pipeline
# ---------------------------------------------------------------------------

def groq_chat(prompt, temperature=0.8, max_tokens=1800):
    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()


def generate_story(topic, num_scenes):
    total_seconds = num_scenes * SCENE_SECONDS
    log(f"Picking a topic and writing a factual script (~{total_seconds}s, {num_scenes} scenes)...")
    prompt = f"""You are a history channel scriptwriter. Your channel's niche/topic focus: {topic}

Pick ONE specific, genuinely interesting REAL historical topic or event within that focus.
Every fact you include must be historically accurate - do not invent or exaggerate details.
Write a punchy, cinematic narration script about it, split into exactly {num_scenes} scenes
of roughly equal length (about {SCENE_SECONDS} seconds of spoken narration each, so aim for
roughly {int(total_seconds * 2.3)} words total across all scenes combined).

For each scene also write a short visual description (image_prompt) of what should be drawn to
illustrate that part of the narration - concrete, vivid, specific (people, setting, action).

Also return 5-8 relevant hashtags for the video (no # symbol, no spaces, lowercase).

Return ONLY valid JSON, no markdown fences, in this exact shape:
{{
  "title": "viral YouTube title under 60 chars, no clickbait lies",
  "description": "2-3 sentence accurate description",
  "hashtags": ["ancienthistory", "..."],
  "scenes": [
    {{"narration": "...", "image_prompt": "..."}},
    ... exactly {num_scenes} of these ...
  ]
}}"""
    raw = groq_chat(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    if len(data.get("scenes", [])) != num_scenes:
        raise ValueError(f"Expected {num_scenes} scenes, got {len(data.get('scenes', []))}")
    log(f"  Topic: {data['title']}")
    return data


def generate_scene_image(image_prompt, index, num_scenes, orientation):
    log(f"  Generating image {index + 1}/{num_scenes}...")
    dims = ORIENTATIONS[orientation]
    full_prompt = (
        f"A cinematic illustration, {dims['aspect_text']} composition, no text or watermarks, "
        f"{STYLE_SUFFIX}: {image_prompt}"
    )
    img_path = os.path.join(WORK_DIR, f"scene_{index}.png")

    last_err = None
    for provider in HF_IMAGE_PROVIDERS:
        try:
            client = InferenceClient(provider=provider, api_key=HF_TOKEN)
            image = client.text_to_image(full_prompt, model=HF_IMAGE_MODEL)
            image.save(img_path)
            return img_path
        except Exception as e:
            last_err = e
            log(f"    provider '{provider}' failed ({e}), trying next...")

    try:
        log("    all HF providers failed, trying pollinations...")
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(full_prompt)}"
        params = {"width": dims["w"], "height": dims["h"], "nologo": "true", "model": "flux"}
        if POLLINATIONS_TOKEN:
            params["token"] = POLLINATIONS_TOKEN
        res = requests.get(url, params=params, timeout=90)
        res.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(res.content)
        return img_path
    except Exception as e:
        last_err = e
        log(f"    provider 'pollinations' failed ({e})")

    raise RuntimeError(f"All image providers failed. Last error: {last_err}")


def image_to_clip(img_path, index, seconds, orientation):
    dims = ORIENTATIONS[orientation]
    w, h = dims["w"], dims["h"]
    clip_path = os.path.join(WORK_DIR, f"clip_{index}.mp4")
    frames = int(25 * seconds)
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", img_path,
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
               f"zoompan=z='min(zoom+0.0018,1.4)':d={frames}:s={w}x{h}:fps=25",
        "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
        "-pix_fmt", "yuv420p", clip_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return clip_path


def concat_clips(clip_paths):
    log("Stitching scenes together...")
    list_path = os.path.join(WORK_DIR, "concat_list.txt")
    with open(list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    combined_path = os.path.join(WORK_DIR, "combined.mp4")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", combined_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    return combined_path


def generate_narration(full_script):
    log("Generating narration voiceover (Fish Audio)...")
    if not FISH_AUDIO_API_KEY:
        log("  No FISH_AUDIO_API_KEY set - video will have no narration")
        return None

    payload = {
        "text": full_script,
        "format": "mp3",
        "mp3_bitrate": 128,
        "normalize": True,
        "chunk_length": 300,
    }
    if FISH_AUDIO_VOICE_ID:
        payload["reference_id"] = FISH_AUDIO_VOICE_ID

    try:
        res = requests.post(
            "https://api.fish.audio/v1/tts",
            headers={
                "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
                "Content-Type": "application/json",
                "model": FISH_AUDIO_MODEL,
            },
            json=payload,
            timeout=120,
        )
        if res.status_code == 200:
            audio_path = os.path.join(WORK_DIR, "narration.mp3")
            with open(audio_path, "wb") as f:
                f.write(res.content)
            return audio_path
        log(f"  Fish Audio error {res.status_code}: {res.text[:300]}")
        return None
    except Exception as e:
        log(f"  Fish Audio failed: {e}")
        return None


def get_background_music():
    if not BACKGROUND_MUSIC_URL:
        return None
    try:
        log("Downloading background music...")
        res = requests.get(BACKGROUND_MUSIC_URL, timeout=60)
        res.raise_for_status()
        music_path = os.path.join(WORK_DIR, "music.mp3")
        with open(music_path, "wb") as f:
            f.write(res.content)
        return music_path
    except Exception as e:
        log(f"  Background music download failed: {e}")
        return None


def mux_narration(video_path, audio_path, output_path):
    log("Adding narration to video...")
    music_path = get_background_music()

    if not audio_path and not music_path:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-c", "copy", output_path]
    elif audio_path and not music_path:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", output_path,
        ]
    elif music_path and not audio_path:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-stream_loop", "-1", "-i", music_path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
            "-filter:a", f"volume={BACKGROUND_MUSIC_VOLUME}", "-shortest", output_path,
        ]
    else:
        log(f"  Mixing narration + background music (music at {BACKGROUND_MUSIC_VOLUME}x volume)...")
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-stream_loop", "-1", "-i", music_path,
            "-filter_complex",
            f"[1:a]volume=1.0[narr];[2:a]volume={BACKGROUND_MUSIC_VOLUME}[music];"
            f"[narr][music]amix=inputs=2:duration=first:dropout_transition=0[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-shortest", output_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    return output_path


def build_hashtags(story, orientation):
    tags = story.get("hashtags", []) or []
    clean = []
    for t in tags:
        t = "".join(ch for ch in t if ch.isalnum())
        if t:
            clean.append(f"#{t}")
    if orientation == "shorts":
        clean.append("#Shorts")
    # de-dupe, keep order
    seen = set()
    out = []
    for t in clean:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return " ".join(out)


def upload_to_youtube(file_path, title, description, orientation):
    log("Uploading to YouTube...")
    try:
        creds = get_google_creds(["https://www.googleapis.com/auth/youtube.upload"])
        creds.refresh(Request())
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {"title": title[:100], "description": description, "categoryId": "27"},
            "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
        }
        media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
        req = youtube.videos().insert(part=",".join(body.keys()), body=body, media_body=media)
        response = None
        while response is None:
            status, response = req.next_chunk()
        vid = response.get("id")
        log(f"  Live -> https://youtube.com/watch?v={vid}")
        return vid
    except Exception as e:
        log(f"  Upload failed: {e}")
        return None


def run_pipeline():
    global pipeline_running
    with pipeline_state_lock:
        if pipeline_running:
            log("Pipeline already running - ignoring this trigger.")
            return
        pipeline_running = True

    log("\nStarting new history video...")
    clip_paths, combined, narration_path, final_path = [], None, None, None
    try:
        cfg = get_config()
        orientation = cfg["orientation"]
        num_scenes = max(1, round(cfg["duration_seconds"] / SCENE_SECONDS))

        story = generate_story(cfg["topic"], num_scenes)
        full_narration = " ".join(s["narration"] for s in story["scenes"])

        for i, scene in enumerate(story["scenes"]):
            img_path = generate_scene_image(scene["image_prompt"], i, num_scenes, orientation)
            clip_path = image_to_clip(img_path, i, SCENE_SECONDS, orientation)
            clip_paths.append(clip_path)

        combined = concat_clips(clip_paths)
        narration_path = generate_narration(full_narration)
        final_path = os.path.join(WORK_DIR, "final_history.mp4")
        mux_narration(combined, narration_path, final_path)

        hashtags = build_hashtags(story, orientation)
        description = f"{story['description']}\n\n{hashtags}".strip()

        vid = upload_to_youtube(final_path, story["title"], description, orientation)
        if vid:
            with open(DONE_FILE, "a") as f:
                f.write(f"{time.time()} | {story['title']}\n")
            log("Pipeline complete!")
        else:
            log("Upload failed - pipeline stopped")

    except Exception as e:
        log(f"Pipeline error: {e}")
    finally:
        for p in clip_paths + [combined, narration_path, final_path]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        with pipeline_state_lock:
            pipeline_running = False


def autopilot_loop():
    wait_seconds = STARTUP_WAIT_HOURS * 3600
    log(f"Startup window: waiting {STARTUP_WAIT_HOURS}h before the first unattended run. "
        f"Visit the site to configure the topic/duration/orientation, or click "
        f"'Generate & Post Now' to skip the wait.")
    while True:
        next_run_at[0] = time.time() + wait_seconds
        triggered = trigger_event.wait(timeout=wait_seconds)
        trigger_event.clear()
        run_pipeline()
        wait_seconds = AUTOPILOT_INTERVAL_HOURS * 3600
        log(f"Sleeping {AUTOPILOT_INTERVAL_HOURS}h until next autopilot run "
            f"(or trigger manually anytime)...")


def main():
    threading.Thread(target=start_server, daemon=True).start()
    threading.Thread(target=autopilot_loop, daemon=True).start()
    log("Bot started.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
