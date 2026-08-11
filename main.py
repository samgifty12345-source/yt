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
# reference_id of a voice from your Fish Audio account (or a public model id).
# Leave blank to use the model's default voice.
FISH_AUDIO_VOICE_ID = os.environ.get("FISH_AUDIO_VOICE_ID", "")
# "s2.1-pro-free" is free; "s2.1-pro" is the paid/higher-latency-priority tier.
FISH_AUDIO_MODEL = os.environ.get("FISH_AUDIO_MODEL", "s2.1-pro-free")

# Edit this anytime in your host's env var settings to change the channel's niche.
# No code changes or redeploy of code needed - just update the variable.
NICHE_PROMPT = os.environ.get(
    "NICHE_PROMPT",
    "Ancient history and forgotten historical events, told dramatically but 100% factually accurate."
)

# Consistency style baked into every image prompt since HF Inference has no
# reference-image conditioning like Gemini did.
STYLE_SUFFIX = os.environ.get(
    "STYLE_SUFFIX",
    "flat 2D cartoon illustration style, muted earth tones, thick black outlines, consistent character design"
)

# How often the bot generates + posts a video on its own, fully unattended.
AUTOPILOT_INTERVAL_HOURS = float(os.environ.get("AUTOPILOT_INTERVAL_HOURS", "24"))

# Free-tier Hugging Face model. Leave unset to use the default.
HF_IMAGE_MODEL = os.environ.get("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")

# Ordered list of HF Inference Providers to try for image generation. HF has
# been reshuffling which providers serve which models (hf-inference dropped
# FLUX.1-schnell entirely in July 2026 with a 410), so we fall through a list
# instead of hardcoding one provider. Override via env if needed, comma-separated.
HF_IMAGE_PROVIDERS = [
    p.strip() for p in os.environ.get("HF_IMAGE_PROVIDERS", "nscale,fal-ai,hf-inference").split(",")
    if p.strip()
]

# Pollinations (https://pollinations.ai) - free, key-less last-resort image
# fallback used only when every HF provider above fails (out of credits,
# deprecated model, etc). Set POLLINATIONS_TOKEN if you register a free
# account - raises the anonymous rate limit and removes the watermark more
# reliably than the "nologo" query param alone.
POLLINATIONS_TOKEN = os.environ.get("POLLINATIONS_TOKEN", "")

NUM_SCENES = 6          # 6 scenes x 10s = 60s video
SCENE_SECONDS = 10

pipeline_log = ["History bot ready. Waiting for first autopilot run or manual trigger."]
lock = threading.Lock()

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>AI History Shorts Bot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #0f0f0f; color: #fff; padding: 30px; }}
    h1 {{ color: #ff0000; margin-bottom: 20px; }}
    .card {{ background: #1a1a1a; border-radius: 12px; padding: 24px; max-width: 600px; }}
    .status {{ padding: 12px; background: #2a2a2a; border-radius: 8px; font-size: 14px; color: #aaa; margin-bottom: 16px; }}
    .log {{ background: #111; border-radius: 8px; padding: 16px; font-size: 12px; color: #0f0; font-family: monospace; max-height: 400px; overflow-y: auto; margin-bottom: 16px; white-space: pre-wrap; }}
    button {{ background: #ff0000; color: #fff; border: none; padding: 12px 28px; border-radius: 8px; font-size: 16px; cursor: pointer; }}
    button:hover {{ background: #cc0000; }}
    .niche {{ color: #888; font-size: 13px; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>AI History Shorts Bot</h1>
  <div class="card">
    <div class="status">Posted: <b>{done_count}</b> | Autopilot every <b>{interval}h</b></div>
    <div class="niche">Niche: {niche}</div>
    <h3 style="color:#aaa;font-size:14px;margin-bottom:10px;">PIPELINE STATUS</h3>
    <div class="log">{log_content}</div>
    <form method="POST" action="/trigger">
      <button type="submit">Generate & Post Now</button>
    </form>
  </div>
</body>
</html>"""


def log(msg):
    print(msg, flush=True)
    with lock:
        pipeline_log.append(msg)
        if len(pipeline_log) > 80:
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
        with lock:
            log_html = "\n".join(pipeline_log[-40:])
        html = HTML.format(
            done_count=done_count,
            interval=AUTOPILOT_INTERVAL_HOURS,
            niche=NICHE_PROMPT,
            log_content=log_html,
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        threading.Thread(target=run_pipeline, daemon=True).start()
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, *args):
        pass


def start_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


def groq_chat(prompt, temperature=0.8, max_tokens=1200):
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


def generate_story():
    log("Picking a topic and writing a factual script...")
    prompt = f"""You are a history channel scriptwriter. Your channel's niche: {NICHE_PROMPT}

Pick ONE specific, genuinely interesting REAL historical topic or event within that niche.
Every fact you include must be historically accurate - do not invent or exaggerate details.
Write a punchy, cinematic ~130-150 word narration script about it, split into exactly {NUM_SCENES} scenes
of roughly equal length (about {SCENE_SECONDS} seconds of spoken narration each).

For each scene also write a short visual description (image_prompt) of what should be drawn to
illustrate that part of the narration - concrete, vivid, specific (people, setting, action).

Return ONLY valid JSON, no markdown fences, in this exact shape:
{{
  "title": "viral YouTube title under 60 chars, no clickbait lies",
  "description": "2-3 sentence accurate description",
  "scenes": [
    {{"narration": "...", "image_prompt": "..."}},
    ... exactly {NUM_SCENES} of these ...
  ]
}}"""
    raw = groq_chat(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    if len(data.get("scenes", [])) != NUM_SCENES:
        raise ValueError(f"Expected {NUM_SCENES} scenes, got {len(data.get('scenes', []))}")
    log(f"  Topic: {data['title']}")
    return data


def generate_scene_image(image_prompt, index):
    log(f"  Generating image {index + 1}/{NUM_SCENES}...")
    full_prompt = (
        f"A cinematic illustration, vertical 9:16 composition, no text or watermarks, "
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

    # Last resort: Pollinations - free, key-less, no HF credits needed.
    try:
        log("    all HF providers failed, trying pollinations...")
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(full_prompt)}"
        params = {"width": 1080, "height": 1920, "nologo": "true", "model": "flux"}
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


def image_to_clip(img_path, index, seconds):
    clip_path = os.path.join(WORK_DIR, f"clip_{index}.mp4")
    frames = int(25 * seconds)
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", img_path,
        "-vf", f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
               f"zoompan=z='min(zoom+0.0018,1.4)':d={frames}:s=1080x1920:fps=25",
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
    """Fish Audio TTS. https://docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech"""
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


def mux_narration(video_path, audio_path, output_path):
    log("Adding narration to video...")
    if not audio_path:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-c", "copy", output_path]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", output_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=90)
    return output_path


def upload_to_youtube(file_path, title, description):
    log("Uploading to YouTube...")
    try:
        creds = get_google_creds(["https://www.googleapis.com/auth/youtube.upload"])
        creds.refresh(Request())
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {"title": title[:100], "description": f"{description}\n\n#shorts #history",
                        "categoryId": "27"},
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
    log("\nStarting new history short...")
    clip_paths, combined, narration_path, final_path = [], None, None, None
    try:
        story = generate_story()
        full_narration = " ".join(s["narration"] for s in story["scenes"])

        for i, scene in enumerate(story["scenes"]):
            img_path = generate_scene_image(scene["image_prompt"], i)
            clip_path = image_to_clip(img_path, i, SCENE_SECONDS)
            clip_paths.append(clip_path)

        combined = concat_clips(clip_paths)
        narration_path = generate_narration(full_narration)
        final_path = os.path.join(WORK_DIR, "final_history.mp4")
        mux_narration(combined, narration_path, final_path)

        vid = upload_to_youtube(final_path, story["title"], story["description"])
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


def autopilot_loop():
    while True:
        run_pipeline()
        log(f"Sleeping {AUTOPILOT_INTERVAL_HOURS}h until next autopilot run...")
        time.sleep(AUTOPILOT_INTERVAL_HOURS * 3600)


def main():
    threading.Thread(target=start_server, daemon=True).start()
    threading.Thread(target=autopilot_loop, daemon=True).start()
    log("Bot started. First autopilot run begins immediately, then repeats on schedule.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
