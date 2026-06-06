"""
AI Video Pipeline — Viral Clip Edition
1. You paste a YouTube link
2. yt-dlp downloads the video + auto-captions
3. Groq reads captions and picks the best 60-second moment
4. ffmpeg clips that section
5. Groq writes a commentary hook ("You won't believe what happened...")
6. ElevenLabs voices the commentary
7. ffmpeg mixes commentary over original audio (ducked)
8. Uploads to YouTube as a Short
"""

import os
import time
import json
import tempfile
import requests
import subprocess
import threading
import re
from http.server import HTTPServer, BaseHTTPRequestHandler

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

WORK_DIR     = tempfile.gettempdir()
DONE_FILE    = "done_pipeline.txt"

GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

pipeline_log = ["Pipeline ready... Waiting for a YouTube link."]
pending_url  = [None]
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
  </style>
</head>
<body>
  <h1>🤖 AI Clip Pipeline</h1>
  <div class="card">
    <div class="status">✅ Posted: <b>{done_count}</b></div>
    <h3 style="color:#aaa;font-size:14px;margin-bottom:10px;">PIPELINE STATUS</h3>
    <div class="log">{log_content}</div>
    <form method="POST" action="/trigger">
      <input name="url" placeholder="Paste your YouTube video link here..." required>
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


def download_video_and_captions(url):
    log("📥 Downloading video + captions...")
    video_path = os.path.join(WORK_DIR, "source_video.mp4")
    captions_path = os.path.join(WORK_DIR, "captions.json")

    # Download captions first
    cap_cmd = [
        "yt-dlp",
        "--write-auto-sub", "--sub-lang", "en",
        "--sub-format", "json3",
        "--skip-download",
        "-o", os.path.join(WORK_DIR, "captions"),
        url
    ]
    try:
        subprocess.run(cap_cmd, check=True, capture_output=True, timeout=60)
        log("  ✅ Captions downloaded")
    except Exception as e:
        log(f"  ⚠️ Caption download issue: {e}")

    # Download video (max 1080p, mp4)
    vid_cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", video_path,
        url
    ]
    try:
        subprocess.run(vid_cmd, check=True, capture_output=True, timeout=300)
        log(f"  ✅ Video downloaded")
        return video_path
    except Exception as e:
        log(f"  ❌ Video download failed: {e}")
        return None


def parse_captions():
    """Find and parse the downloaded caption file, return list of {start, end, text}"""
    import glob
    files = glob.glob(os.path.join(WORK_DIR, "captions*.json3")) + \
            glob.glob(os.path.join(WORK_DIR, "captions*.vtt")) + \
            glob.glob(os.path.join(WORK_DIR, "captions*.json"))
    
    if not files:
        log("  ⚠️ No caption file found")
        return []

    cap_file = files[0]
    log(f"  📄 Parsing: {os.path.basename(cap_file)}")

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
            log(f"  ⚠️ json3 parse error: {e}")

    elif cap_file.endswith(".vtt"):
        try:
            with open(cap_file) as f:
                content = f.read()
            pattern = r'(\d+:\d+:\d+\.\d+) --> (\d+:\d+:\d+\.\d+).*?\n(.*?)(?=\n\n|\Z)'
            for m in re.finditer(pattern, content, re.DOTALL):
                def ts(t):
                    parts = t.replace(",", ".").split(":")
                    return sum(float(x) * 60**i for i, x in enumerate(reversed(parts)))
                text = re.sub(r'<[^>]+>', '', m.group(3)).strip()
                if text:
                    segments.append({"start": ts(m.group(1)), "end": ts(m.group(2)), "text": text})
        except Exception as e:
            log(f"  ⚠️ vtt parse error: {e}")

    log(f"  ✅ Parsed {len(segments)} caption segments")
    return segments


def find_best_clip(segments):
    """Ask Groq to find the most engaging 60-second window"""
    log("🧠 Finding best 60-second moment...")

    if not segments:
        log("  ⚠️ No captions — using first 60 seconds")
        return 0, 60, "Incredible moment from this video"

    # Build transcript with timestamps
    transcript_lines = []
    for s in segments[:300]:  # First 300 segments max
        transcript_lines.append(f"[{s['start']:.1f}s] {s['text']}")
    transcript = "\n".join(transcript_lines)

    prompt = f"""You are a YouTube Shorts editor. Find the single most engaging, surprising, or emotional 60-second moment in this transcript.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON:
{{
  "start_seconds": <number - start time in seconds>,
  "end_seconds": <number - exactly 60 seconds after start>,
  "hook": "one punchy sentence teasing what happens (e.g. 'You won't believe what happened next...')",
  "title": "short viral YouTube title under 60 chars",
  "description": "2-3 sentence description"
}}

Pick the moment with the most drama, surprise, or emotion. start_seconds and end_seconds must be exactly 60 seconds apart."""

    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7},
            timeout=30
        )
        res.raise_for_status()
        text = res.json()["choices"][0]["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        start = float(data["start_seconds"])
        end = float(data["end_seconds"])
        hook = data.get("hook", "You won't believe this...")
        log(f"  ✅ Best moment: {start:.0f}s - {end:.0f}s")
        log(f"  🎣 Hook: {hook}")
        return start, end, data
    except Exception as e:
        log(f"  ⚠️ AI selection failed ({e}) — using first 60s")
        return 0, 60, {"hook": "You won't believe this...", "title": "Incredible Moment", "description": "Watch this amazing clip."}


def clip_video(video_path, start, end):
    log(f"✂️ Clipping {start:.0f}s - {end:.0f}s...")
    clip_path = os.path.join(WORK_DIR, "clip.mp4")
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        clip_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        log("  ✅ Clip created")
        return clip_path
    except Exception as e:
        log(f"  ❌ Clip failed: {e}")
        return None


def generate_commentary(hook, segments, start, end):
    log("📝 Generating commentary script...")
    
    # Get the transcript for this window
    window_text = " ".join(
        s["text"] for s in segments if start <= s["start"] <= end
    )

    prompt = f"""Write a short, punchy 15-second spoken commentary for a YouTube Short.

The clip is about: {hook}
What happens in the clip: {window_text[:500]}

Rules:
- Start with a hook like "You won't believe..." or "Wait till you see..."
- Maximum 40 words
- Conversational, excited tone
- End with a cliffhanger or reaction
- No hashtags, no emojis

Return ONLY the spoken script text, nothing else."""

    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.9},
            timeout=30
        )
        res.raise_for_status()
        script = res.json()["choices"][0]["message"]["content"].strip()
        log(f"  ✅ Script: {script[:80]}...")
        return script
    except Exception as e:
        log(f"  ⚠️ Commentary failed: {e}")
        return hook


def generate_voiceover(script):
    log("🎙️ Generating voiceover...")
    if not ELEVENLABS_API_KEY:
        log("  ⚠️ No ElevenLabs key — skipping voiceover")
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
        else:
            log(f"  ❌ ElevenLabs error: {res.status_code}")
            return None
    except Exception as e:
        log(f"  ❌ Voiceover failed: {e}")
        return None


def mix_audio(clip_path, voiceover_path, output_path):
    log("🎚️ Mixing audio...")
    if not voiceover_path:
        # No voiceover — just copy clip
        cmd = ["ffmpeg", "-y", "-i", clip_path, "-c", "copy", output_path]
    else:
        # Duck original audio to 20%, commentary at 100%
        cmd = [
            "ffmpeg", "-y",
            "-i", clip_path,
            "-i", voiceover_path,
            "-filter_complex",
            "[0:a]volume=0.2[a1];[1:a]volume=1.0[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            output_path
        ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        log("  ✅ Audio mixed")
        return output_path
    except Exception as e:
        log(f"  ❌ Mix failed: {e}")
        return clip_path  # fallback to unmixed clip


def upload_to_youtube(file_path, title, description):
    log("📤 Uploading to YouTube...")
    try:
        creds = Credentials(
            token=None,
            refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["YOUTUBE_CLIENT_ID"],
            client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
            scopes=["https://www.googleapis.com/auth/youtube.upload"],
        )
        creds.refresh(Request())
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title": title[:100],
                "description": f"{description}\n\n#shorts #viral #fyp",
                "categoryId": "22"
            },
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
        log(f"  ✅ Uploaded → https://youtube.com/watch?v={vid}")
        return vid
    except Exception as e:
        log(f"  ❌ Upload failed: {e}")
        return None


def run_pipeline(url):
    log(f"\n🚀 Starting pipeline for: {url}")

    # 1. Download
    video_path = download_video_and_captions(url)
    if not video_path:
        log("❌ Aborted: download failed")
        return

    # 2. Parse captions
    segments = parse_captions()

    # 3. Find best clip
    start, end, clip_data = find_best_clip(segments)
    if isinstance(clip_data, dict):
        hook = clip_data.get("hook", "You won't believe this...")
        title = clip_data.get("title", "Incredible Moment")
        description = clip_data.get("description", "Watch this amazing clip.")
    else:
        hook, title, description = str(clip_data), "Incredible Moment", "Watch this."

    # 4. Clip video
    clip_path = clip_video(video_path, start, end)
    if not clip_path:
        log("❌ Aborted: clip failed")
        return

    # 5. Generate commentary
    script = generate_commentary(hook, segments, start, end)

    # 6. Voiceover
    voiceover_path = generate_voiceover(script)

    # 7. Mix
    final_path = os.path.join(WORK_DIR, "final_short.mp4")
    result = mix_audio(clip_path, voiceover_path, final_path)

    # 8. Upload
    vid = upload_to_youtube(final_path, title, description)
    if vid:
        with open(DONE_FILE, "a") as f:
            f.write(f"{time.time()} | {title}\n")
        log("✅ Pipeline complete!\n")
    else:
        log("❌ Upload failed")

    # Cleanup
    for path in [video_path, clip_path, voiceover_path, final_path]:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except:
            pass


def main():
    threading.Thread(target=start_server, daemon=True).start()
    log("🤖 Bot started. Paste a YouTube link to begin.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
