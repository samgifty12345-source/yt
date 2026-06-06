"""
AI Video Pipeline — Full Automated Edition
1. Gemini generates a story + 6 scene descriptions
2. Gemini Imagen generates 1 image per scene
3. ffmpeg applies Ken Burns zoom/pan effect to each image
4. ElevenLabs generates voiceover
5. ffmpeg combines all clips + audio
6. YouTube API uploads the final video
"""

import os
import time
import json
import tempfile
import requests
import subprocess
import threading
import random
from http.server import HTTPServer, BaseHTTPRequestHandler

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google import genai

WORK_DIR      = tempfile.gettempdir()
DONE_FILE     = "done_pipeline.txt"
WAIT_SECONDS  = 24 * 3600

GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

last_post_time = [None]
lock = threading.Lock()

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>AI Video Pipeline</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #0f0f0f; color: #fff; padding: 30px; }}
    h1 {{ color: #ff0000; margin-bottom: 20px; }}
    .card {{ background: #1a1a1a; border-radius: 12px; padding: 24px; max-width: 600px; }}
    .status {{ padding: 12px; background: #2a2a2a; border-radius: 8px; font-size: 14px; color: #aaa; margin-bottom: 16px; }}
    .log {{ background: #111; border-radius: 8px; padding: 16px; font-size: 12px; color: #0f0; font-family: monospace; max-height: 300px; overflow-y: auto; }}
    h3 {{ margin-bottom: 10px; color: #aaa; font-size: 14px; }}
    button {{ background: #ff0000; color: #fff; border: none; padding: 12px 28px;
      border-radius: 8px; font-size: 16px; cursor: pointer; margin-top: 16px; }}
    button:hover {{ background: #cc0000; }}
  </style>
</head>
<body>
  <h1>🤖 AI Video Pipeline</h1>
  <div class="card">
    <div class="status">
      ✅ Posted: <b>{done_count}</b> &nbsp;|&nbsp;
      ⏱ Next video in: <b>{next_post}</b>
    </div>
    <h3>PIPELINE STATUS</h3>
    <div class="log">{log_content}</div>
    <form method="POST" action="/trigger">
      <button type="submit">▶️ Generate & Post Now</button>
    </form>
  </div>
</body>
</html>"""

pipeline_log = ["Pipeline ready..."]


def log(msg):
    print(msg)
    pipeline_log.append(msg)
    if len(pipeline_log) > 50:
        pipeline_log.pop(0)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        done_count = 0
        if os.path.exists(DONE_FILE):
            with open(DONE_FILE) as f:
                done_count = len([l for l in f if l.strip()])
        if last_post_time[0] is None:
            next_post = "soon"
        else:
            elapsed = time.time() - last_post_time[0]
            remaining = max(0, WAIT_SECONDS - elapsed)
            hrs  = int(remaining // 3600)
            mins = int((remaining % 3600) // 60)
            next_post = f"{hrs}h {mins}m"
        log_html = "<br>".join(pipeline_log[-30:])
        html = HTML.format(done_count=done_count, next_post=next_post, log_content=log_html)
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


def generate_story():
    log("📝 Generating story...")
    topics = [
        "a surprising ancient history fact",
        "a mind-blowing science discovery",
        "a fascinating true crime story",
        "an unbelievable survival story",
        "a weird but true historical event",
        "a shocking space discovery",
        "a little-known historical figure who changed the world",
        "a strange natural phenomenon explained",
        "a bizarre world record that actually exists",
        "an incredible animal behavior fact",
        "a little known fact about ancient egypt",
        "a shocking fact about the human body",
        "an unbelievable coincidence in history",
        "a mysterious unsolved historical event",
    ]
    topic = random.choice(topics)
    log(f"  Topic: {topic}")

    prompt = f"""Write a short, engaging 60-second narration script about: {topic}

Split it into exactly 6 scenes (each ~10 seconds when read aloud).

Return ONLY valid JSON, no extra text:
{{
  "title": "catchy YouTube title under 70 chars",
  "description": "2-3 sentence description for YouTube",
  "hashtags": "#history #facts #viral #shorts #fyp #trending",
  "narration": "full narration script (60 seconds)",
  "scenes": [
    {{"scene": 1, "text": "narration for this scene", "image_prompt": "detailed visual description for AI image generation, cinematic, realistic"}},
    {{"scene": 2, "text": "...", "image_prompt": "..."}},
    {{"scene": 3, "text": "...", "image_prompt": "..."}},
    {{"scene": 4, "text": "...", "image_prompt": "..."}},
    {{"scene": 5, "text": "...", "image_prompt": "..."}},
    {{"scene": 6, "text": "...", "image_prompt": "..."}}
  ]
}}"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        log(f"  ✅ Title: {data['title']}")
        return data
    except Exception as e:
        log(f"  ❌ Story failed: {e}")
        return None


def generate_image(prompt, index):
    log(f"  🎨 Generating image {index+1}/6...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        result = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=f"{prompt}, cinematic, high quality, detailed, 16:9 aspect ratio",
            config={"number_of_images": 1, "aspect_ratio": "16:9"},
        )
        img_path = os.path.join(WORK_DIR, f"scene_{index}.png")
        result.generated_images[0].image.save(img_path)
        log(f"  ✅ Image {index+1} saved")
        return img_path
    except Exception as e:
        log(f"  ❌ Image {index+1} failed: {e}")
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (1280, 720), color=(20, 20, 40))
            draw = ImageDraw.Draw(img)
            draw.text((640, 360), f"Scene {index+1}", fill=(200, 200, 200), anchor="mm")
            img_path = os.path.join(WORK_DIR, f"scene_{index}.png")
            img.save(img_path)
            return img_path
        except:
            return None


def image_to_video(img_path, output_path, duration=10, index=0):
    log(f"  🎬 Creating clip {index+1}/6...")
    effects = [
        "zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=250:s=1280x720",
        "zoompan=z='if(lte(zoom,1.0),1.5,max(1.001,zoom-0.0015))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=250:s=1280x720",
        "zoompan=z='1.3':x='if(lte(on,1),0,x+1.5)':y='ih/2-(ih/zoom/2)':d=250:s=1280x720",
        "zoompan=z='1.3':x='if(lte(on,1),iw,x-1.5)':y='ih/2-(ih/zoom/2)':d=250:s=1280x720",
        "zoompan=z='min(zoom+0.0015,1.5)':x='0':y='0':d=250:s=1280x720",
        "zoompan=z='min(zoom+0.0015,1.5)':x='iw-(iw/zoom)':y='ih-(ih/zoom)':d=250:s=1280x720",
    ]
    effect = effects[index % len(effects)]
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", img_path,
        "-vf", effect, "-t", str(duration), "-r", "25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log(f"  ✅ Clip {index+1} created")
        return output_path
    except Exception as e:
        log(f"  ❌ Clip {index+1} failed: {e}")
        return None


def generate_voiceover(narration):
    log("🎙️ Generating voiceover...")
    try:
        res = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={
                "text": narration,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
            },
            timeout=60
        )
        if res.status_code == 200:
            audio_path = os.path.join(WORK_DIR, "voiceover.mp3")
            with open(audio_path, "wb") as f:
                f.write(res.content)
            log("  ✅ Voiceover generated")
            return audio_path
        else:
            log(f"  ❌ ElevenLabs error: {res.status_code} {res.text}")
            return None
    except Exception as e:
        log(f"  ❌ Voiceover failed: {e}")
        return None


def combine_clips(clip_paths, audio_path, output_path):
    log("🎞️ Combining clips...")
    concat_file = os.path.join(WORK_DIR, "concat.txt")
    with open(concat_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")

    combined_video = os.path.join(WORK_DIR, "combined_video.mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file, "-c", "copy", combined_video
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:
        log(f"  ❌ Concat failed: {e}")
        return None

    if audio_path and os.path.exists(audio_path):
        cmd2 = [
            "ffmpeg", "-y", "-i", combined_video, "-i", audio_path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy",
            "-c:a", "aac", "-shortest", output_path
        ]
    else:
        cmd2 = ["ffmpeg", "-y", "-i", combined_video, "-c", "copy", output_path]

    try:
        subprocess.run(cmd2, check=True, capture_output=True)
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", output_path],
                capture_output=True, text=True
            )
            duration = float(json.loads(result.stdout)["format"]["duration"])
            log(f"  ✅ Final video: {duration:.1f} seconds")
        except:
            log("  ✅ Final video created")
        return output_path
    except Exception as e:
        log(f"  ❌ Audio merge failed: {e}")
        return None


def get_youtube_client():
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(file_path, title, description, hashtags):
    log("📤 Uploading to YouTube...")
    try:
        youtube = get_youtube_client()
        body = {
            "snippet": {
                "title": title[:100],
                "description": f"{description}\n\n{hashtags}",
                "categoryId": "27"
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

        body["snippet"]["title"] = f"{title[:94]} #Shorts"
        media2 = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
        req2 = youtube.videos().insert(part=",".join(body.keys()), body=body, media_body=media2)
        response2 = None
        while response2 is None:
            _, response2 = req2.next_chunk()
        vid2 = response2.get("id")
        log(f"  ✅ Short → https://youtube.com/watch?v={vid2}")
        return vid
    except Exception as e:
        log(f"  ❌ Upload failed: {e}")
        return None


def run_pipeline():
    log("\n🚀 Starting pipeline...")

    story = generate_story()
    if not story:
        log("❌ Aborted: story failed")
        return

    images = []
    for i, scene in enumerate(story["scenes"]):
        img = generate_image(scene["image_prompt"], i)
        images.append(img)
        time.sleep(2)

    clips = []
    for i, img_path in enumerate(images):
        if img_path:
            clip_path = os.path.join(WORK_DIR, f"clip_{i}.mp4")
            clip = image_to_video(img_path, clip_path, duration=10, index=i)
            if clip:
                clips.append(clip)

    if not clips:
        log("❌ Aborted: no clips")
        return

    audio_path = generate_voiceover(story["narration"])

    final_path = os.path.join(WORK_DIR, "final_video.mp4")
    result = combine_clips(clips, audio_path, final_path)
    if not result:
        log("❌ Aborted: combine failed")
        return

    vid = upload_to_youtube(final_path, story["title"], story["description"], story["hashtags"])
    if vid:
        with open(DONE_FILE, "a") as f:
            f.write(f"{time.time()} | {story['title']}\n")
        last_post_time[0] = time.time()
        log("✅ Pipeline complete!\n")
    else:
        log("❌ Upload failed")

    for path in images + clips + [audio_path, final_path]:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except:
            pass


def bot_loop():
    log("🤖 Bot started. Runs once per day.")
    run_pipeline()
    while True:
        if last_post_time[0] is not None:
            elapsed = time.time() - last_post_time[0]
            if elapsed >= WAIT_SECONDS:
                run_pipeline()
        time.sleep(60)


def main():
    threading.Thread(target=start_server, daemon=True).start()
    bot_loop()


if __name__ == "__main__":
    main()
