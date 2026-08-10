import os
import time
import json
import tempfile
import requests
import subprocess
import threading
import re
import glob
from http.server import HTTPServer, BaseHTTPRequestHandler

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

WORK_DIR  = tempfile.gettempdir()
DONE_FILE = "done_pipeline.txt"

GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

COOKIES_PATH = os.path.join(WORK_DIR, "yt_cookies.txt")

YT_COOKIES = os.environ.get("YOUTUBE_COOKIES_TXT", """# Netscape HTTP Cookie File
.youtube.com	TRUE	/	FALSE	1815339931	SID	g.a000-wiVaauPSgjBbb8z2dcCtThe0EBTTQiI6Qc-0nvVjD3wh45zv6iuCyrlq-NyGgstyOM9PwACgYKAW8SARYSFQHGX2Mi2sbMpuVkCSqGFgDfhW0t4BoVAUF8yKotJqkpYRTEiGqXMr2ARXGt0076
.youtube.com	TRUE	/	TRUE	1815339931	__Secure-1PSID	g.a000-wiVaauPSgjBbb8z2dcCtThe0EBTTQiI6Qc-0nvVjD3wh45zT2UejOGIkeuCGh6JDgd5VAACgYKAQASARYSFQHGX2MiC6rz4BjpNf6ZvW30ukcXlRoVAUF8yKpZgaZtZ52zldC5xAYbj2oD0076
.youtube.com	TRUE	/	TRUE	1815339931	SAPISID	t3k8cNGc-g02OGHr/AVACVAUhF0z83og2Q
.youtube.com	TRUE	/	TRUE	1810559899	LOGIN_INFO	AFmmF2swRAIge0gH08i3OiSk5Lx99mbckfZielz-6FORMK7LJ9GmHJMCIBRJYueJg-NIy48J0g_ph4990pFKnDtltjMClocNKU_i:QUQ3MjNmekdGNUE1Qms2Rk90U3VXN3psMktiZlNfbk9kY0pPLWNxajNOZzJmMWJ6NDBVYi1pbmkya3FDeWM2dHJRZW9XZnNaOHlreFFaSUhJdWtoaW8tRlZXeGJrT0VwTWlzcjFUNDNCSnhLZzBtY2hUUWNFSnV6ZFJoRzlwbEI2N2hmWDc3V0FHc1dPelJEaW5wek1sZ1BMQ3R3UnlWMm5GWldqa3ppVVJQZHhlRllPamZfTDVKTzd3UEZWR2Rxa3ZNenJLejNHc3ZpYzRzMHlsS0s2VHBLVFVURmpHQUZwQQ==
""")

with open(COOKIES_PATH, "w") as f:
    f.write(YT_COOKIES)

pipeline_log = ["Pipeline ready... Waiting for a YouTube or Google Drive link."]
lock = threading.Lock()

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>AI Clip Pipeline</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #0f0f0f; color: #fff; padding: 30px; }}
    h1 {{ color: #ff0000; margin-bottom: 20px; }}
    .card {{ background: #1a1a1a; border-radius: 12px; padding: 24px; max-width: 600px; }}
    .status {{ padding: 12px; background: #2a2a2a; border-radius: 8px; font-size: 14px; color: #aaa; margin-bottom: 16px; }}
    .log {{ background: #111; border-radius: 8px; padding: 16px; font-size: 12px; color: #0f0; font-family: monospace; max-height: 300px; overflow-y: auto; margin-bottom: 16px; }}
    input {{ width: 100%; padding: 12px; background: #2a2a2a; border: 1px solid #444; border-radius: 8px; color: #fff; font-size: 14px; margin-bottom: 12px; }}
    button {{ background: #ff0000; color: #fff; border: none; padding: 12px 28px; border-radius: 8px; font-size: 16px; cursor: pointer; }}
    button:hover {{ background: #cc0000; }}
    .hint {{ color: #666; font-size: 12px; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>🤖 AI Clip Pipeline</h1>
  <div class="card">
    <div class="status">✅ Posted: <b>{done_count}</b></div>
    <h3 style="color:#aaa;font-size:14px;margin-bottom:10px;">PIPELINE STATUS</h3>
    <div class="log">{log_content}</div>
    <form method="POST" action="/trigger">
      <input name="url" placeholder="YouTube link OR Google Drive link..." required>
      <p class="hint">Drive: share video → copy link → paste here</p>
      <button type="submit">▶️ Generate & Post Short</button>
    </form>
  </div>
</body>
</html>"""


def log(msg):
    print(msg)
    pipeline_log.append(msg)
    if len(pipeline_log) > 60:
        pipeline_log.pop(0)


def get_google_creds(scopes):
    return Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        scopes=scopes,
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        done_count = 0
        if os.path.exists(DONE_FILE):
            with open(DONE_FILE) as f:
                done_count = len([l for l in f if l.strip()])
        log_html = "<br>".join(pipeline_log[-30:])
        html = HTML.format(done_count=done_count, log_content=log_html)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        url = ""
        for part in body.split("&"):
            if part.startswith("url="):
                url = requests.utils.unquote(part[4:]).strip()
        if url:
            threading.Thread(target=run_pipeline, args=(url,), daemon=True).start()
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, *args):
        pass


def start_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


def is_drive_link(url):
    return "drive.google.com" in url or "docs.google.com" in url


def extract_drive_file_id(url):
    """Extract file ID from various Drive URL formats."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def download_from_drive(url):
    """Download video from Google Drive using OAuth credentials."""
    log("📥 Detected Google Drive link — downloading via Drive API...")
    video_path = os.path.join(WORK_DIR, "source_video.mp4")

    file_id = extract_drive_file_id(url)
    if not file_id:
        log("  ❌ Could not extract file ID from Drive URL")
        return None

    log(f"  📂 File ID: {file_id}")

    try:
        creds = get_google_creds([
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/youtube.upload",
        ])
        creds.refresh(Request())
        drive = build("drive", "v3", credentials=creds)

        # Get file metadata
        meta = drive.files().get(fileId=file_id, fields="name,size,mimeType").execute()
        log(f"  📄 File: {meta.get('name')} ({int(meta.get('size', 0)) // 1024 // 1024}MB)")

        # Download
        request = drive.files().get_media(fileId=file_id)
        with open(video_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    log(f"  ⬇️ {int(status.progress() * 100)}%")

        log("  ✅ Drive video downloaded")
        return video_path

    except Exception as e:
        log(f"  ❌ Drive download failed: {e}")
        return None


def download_from_youtube(url):
    """Try to download from YouTube (may fail on cloud IPs)."""
    log("📥 Downloading from YouTube...")
    video_path = os.path.join(WORK_DIR, "source_video.mp4")

    # Update yt-dlp to nightly
    try:
        subprocess.run(
            ["pip", "install", "yt-dlp", "--pre", "--upgrade", "--break-system-packages", "-q"],
            capture_output=True, timeout=120
        )
        ver = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True).stdout.strip()
        log(f"  yt-dlp → {ver}")
    except Exception as e:
        log(f"  ⚠️ yt-dlp upgrade skipped: {e}")

    fmt = "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best"
    PROXY = "http://snslvrdh:r6ogicxc471x@38.154.203.95:5863"

    attempts = [
        ("web+cookies", ["--cookies", COOKIES_PATH, "--extractor-args", "youtube:player_client=web", "--no-check-certificates", "--no-cache-dir"]),
        ("web+cookies+proxy", ["--cookies", COOKIES_PATH, "--extractor-args", "youtube:player_client=web", "--proxy", PROXY, "--no-check-certificates", "--no-cache-dir"]),
        ("ios (no proxy)", ["--extractor-args", "youtube:player_client=ios", "--no-check-certificates", "--no-cache-dir"]),
        ("ios+proxy", ["--extractor-args", "youtube:player_client=ios", "--proxy", PROXY, "--no-check-certificates", "--no-cache-dir"]),
    ]

    # Captions
    try:
        cap_cmd = ["yt-dlp", "--cookies", COOKIES_PATH, "--extractor-args", "youtube:player_client=web",
                   "--no-check-certificates", "--no-cache-dir",
                   "--write-auto-sub", "--sub-lang", "en", "--sub-format", "json3",
                   "--skip-download", "-o", os.path.join(WORK_DIR, "captions"), url]
        subprocess.run(cap_cmd, capture_output=True, timeout=60)
    except Exception:
        pass

    for label, flags in attempts:
        log(f"  🔄 Trying: {label}...")
        cmd = ["yt-dlp"] + flags + ["-f", fmt, "--merge-output-format", "mp4", "-o", video_path, url]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300, text=True)
            if result.returncode == 0 and os.path.exists(video_path):
                log(f"  ✅ Downloaded ({label})")
                return video_path
            err = result.stderr[-150:].replace("\n", " ")
            log(f"  ⚠️ Failed: ...{err}")
        except subprocess.TimeoutExpired:
            log(f"  ⚠️ Timeout ({label})")
        except Exception as e:
            log(f"  ⚠️ Error ({label}): {e}")

    log("  ❌ All YouTube attempts failed")
    log("  💡 TIP: Upload the video to Google Drive and paste that link instead")
    return None


def download_video_and_captions(url):
    if is_drive_link(url):
        return download_from_drive(url), True   # (path, is_drive)
    else:
        return download_from_youtube(url), False


def parse_captions():
    files = (glob.glob(os.path.join(WORK_DIR, "captions*.json3")) +
             glob.glob(os.path.join(WORK_DIR, "captions*.vtt")))
    if not files:
        return []
    cap_file = files[0]
    log(f"  📄 Parsing captions: {os.path.basename(cap_file)}")
    segments = []
    if cap_file.endswith(".json3"):
        try:
            with open(cap_file) as f:
                data = json.load(f)
            for event in data.get("events", []):
                if "segs" not in event:
                    continue
                start = event.get("tStartMs", 0) / 1000
                dur = event.get("dDurationMs", 2000) / 1000
                text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
                if text and text != "\n":
                    segments.append({"start": start, "end": start + dur, "text": text})
        except Exception as e:
            log(f"  ⚠️ Caption parse error: {e}")
    elif cap_file.endswith(".vtt"):
        try:
            with open(cap_file) as f:
                content = f.read()
            for m in re.finditer(r'(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+).*?\n(.*?)(?=\n\n|\Z)', content, re.DOTALL):
                def ts(t):
                    parts = t.replace(",", ".").split(":")
                    return sum(float(x) * 60**i for i, x in enumerate(reversed(parts)))
                text = re.sub(r'<[^>]+>', '', m.group(3)).strip()
                if text:
                    segments.append({"start": ts(m.group(1)), "end": ts(m.group(2)), "text": text})
        except Exception as e:
            log(f"  ⚠️ VTT parse error: {e}")
    log(f"  ✅ {len(segments)} caption segments")
    return segments


def find_best_clip(segments):
    log("🧠 Finding best 60-second moment...")
    if not segments:
        log("  ⚠️ No captions — using first 60s")
        return 0, 60, {"hook": "You won't believe this...", "title": "Incredible Moment", "description": "Watch this amazing clip."}

    transcript = "\n".join(f"[{s['start']:.1f}s] {s['text']}" for s in segments[:300])
    prompt = f"""You are a YouTube Shorts editor. Find the most engaging 60-second moment.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON:
{{
  "start_seconds": <number>,
  "end_seconds": <number - exactly 60 seconds after start>,
  "hook": "one punchy sentence",
  "title": "viral YouTube title under 60 chars",
  "description": "2-3 sentence description"
}}"""
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7},
            timeout=30
        )
        res.raise_for_status()
        text = res.json()["choices"][0]["message"]["content"].strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        start, end = float(data["start_seconds"]), float(data["end_seconds"])
        log(f"  ✅ {start:.0f}s - {end:.0f}s | {data.get('hook', '')}")
        return start, end, data
    except Exception as e:
        log(f"  ⚠️ AI failed ({e}) — using first 60s")
        return 0, 60, {"hook": "You won't believe this...", "title": "Incredible Moment", "description": "Watch this amazing clip."}


def clip_video(video_path, start, end):
    log(f"✂️ Clipping {start:.0f}s–{end:.0f}s and reformatting to 9:16...")
    clip_path = os.path.join(WORK_DIR, "clip.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", video_path, "-t", str(end - start),
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", "-c:a", "aac",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        clip_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        log("  ✅ Clip created")
        return clip_path
    except Exception as e:
        log(f"  ❌ Clip failed: {e}")
        return None


def generate_commentary(hook, segments, start, end):
    log("📝 Generating commentary...")
    window_text = " ".join(s["text"] for s in segments if start <= s["start"] <= end)
    prompt = f"""Punchy 15-second spoken commentary for a YouTube Short.
About: {hook}
Content: {window_text[:500]}
Rules: max 40 words, hook opener, excited tone, cliffhanger ending, no hashtags/emojis.
Return ONLY the script."""
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.9},
            timeout=30
        )
        res.raise_for_status()
        script = res.json()["choices"][0]["message"]["content"].strip()
        log(f"  ✅ Script ready")
        return script
    except Exception as e:
        log(f"  ⚠️ Commentary failed: {e}")
        return hook


def generate_voiceover(script):
    log("🎙️ Generating voiceover...")
    if not ELEVENLABS_API_KEY:
        log("  ⚠️ No ElevenLabs key — skipping")
        return None
    try:
        res = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": script, "model_id": "eleven_monolingual_v1", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
            timeout=60
        )
        if res.status_code == 200:
            audio_path = os.path.join(WORK_DIR, "commentary.mp3")
            with open(audio_path, "wb") as f:
                f.write(res.content)
            log("  ✅ Voiceover generated")
            return audio_path
        log(f"  ❌ ElevenLabs {res.status_code}")
        return None
    except Exception as e:
        log(f"  ❌ Voiceover failed: {e}")
        return None


def mix_audio(clip_path, voiceover_path, output_path):
    log("🎚️ Mixing audio...")
    if not voiceover_path:
        cmd = ["ffmpeg", "-y", "-i", clip_path, "-c", "copy", output_path]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", clip_path, "-i", voiceover_path,
            "-filter_complex", "[0:a]volume=0.2[a1];[1:a]volume=1.0[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", output_path
        ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        log("  ✅ Mixed")
        return output_path
    except Exception as e:
        log(f"  ❌ Mix failed: {e}")
        return clip_path


def upload_to_youtube(file_path, title, description):
    log("📤 Uploading Short to YouTube...")
    try:
        creds = get_google_creds(["https://www.googleapis.com/auth/youtube.upload"])
        creds.refresh(Request())
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {"title": title[:100], "description": f"{description}\n\n#shorts #viral #fyp", "categoryId": "22"},
            "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
        }
        media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
        req = youtube.videos().insert(part=",".join(body.keys()), body=body, media_body=media)
        response = None
        while response is None:
            status, response = req.next_chunk()
            if status:
                log(f"  ⬆️ {int(status.progress()*100)}%")
        vid = response.get("id")
        log(f"  ✅ Live → https://youtube.com/watch?v={vid}")
        return vid
    except Exception as e:
        log(f"  ❌ Upload failed: {e}")
        return None


def run_pipeline(url):
    log(f"\n🚀 Starting pipeline for: {url}")

    result = download_video_and_captions(url)
    video_path, is_drive = result

    if not video_path:
        log("❌ Aborted: could not get video")
        return

    segments = parse_captions() if not is_drive else []
    if is_drive:
        log("  ℹ️ Drive video — no captions, using first 60s")

    start, end, clip_data = find_best_clip(segments)
    hook        = clip_data.get("hook", "You won't believe this...") if isinstance(clip_data, dict) else str(clip_data)
    title       = clip_data.get("title", "Incredible Moment") if isinstance(clip_data, dict) else "Incredible Moment"
    description = clip_data.get("description", "Watch this.") if isinstance(clip_data, dict) else "Watch this."

    clip_path = clip_video(video_path, start, end)
    if not clip_path:
        log("❌ Aborted: clip failed")
        return

    script = generate_commentary(hook, segments, start, end)
    voiceover_path = generate_voiceover(script)

    final_path = os.path.join(WORK_DIR, "final_short.mp4")
    mix_audio(clip_path, voiceover_path, final_path)

    vid = upload_to_youtube(final_path, title, description)
    if vid:
        with open(DONE_FILE, "a") as f:
            f.write(f"{time.time()} | {title}\n")
        log("✅ Pipeline complete!")
    else:
        log("❌ Upload failed")

    for path in [video_path, clip_path, voiceover_path, final_path]:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except:
            pass


def main():
    threading.Thread(target=start_server, daemon=True).start()
    log("🤖 Bot started. Paste a YouTube or Google Drive link to begin.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
