"""
YouTube Auto Uploader — Web Dashboard + Auto Post Edition
- Posts links submitted via dashboard immediately
- If nothing submitted for 4 hours, auto posts the default video
- Posts as both regular video and Short
- Uses Groq to generate title and description
"""

import os
import csv
import time
import tempfile
import threading
import requests
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

import gdown
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

DONE_FILE    = "done.txt"
DOWNLOAD_DIR = tempfile.gettempdir()
WAIT_SECONDS = 4 * 3600  # 4 hours

DEFAULT_VIDEO = {
    "url": "https://drive.google.com/file/d/1VJhJFJp_gcvpSoriZ9ZzaGjsFRzNK4kl/view?usp=sharing",
    "hint": "ronaldo skills goals edit"
}

pending_links = []
lock = threading.Lock()
last_post_time = [0]  # using list so it's mutable inside threads


# ── Web Dashboard ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
  <title>YouTube Auto Uploader</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; background: #0f0f0f; color: #fff; padding: 30px; }}
    h1 {{ color: #ff0000; margin-bottom: 20px; }}
    .card {{ background: #1a1a1a; border-radius: 12px; padding: 24px; max-width: 600px; }}
    input {{ width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #333;
      background: #2a2a2a; color: #fff; font-size: 15px; margin-bottom: 12px; }}
    button {{ background: #ff0000; color: #fff; border: none; padding: 12px 28px;
      border-radius: 8px; font-size: 16px; cursor: pointer; width: 100%; }}
    button:hover {{ background: #cc0000; }}
    .status {{ margin-top: 20px; padding: 12px; background: #2a2a2a; border-radius: 8px; font-size: 14px; color: #aaa; }}
    .queue {{ margin-top: 16px; }}
    .item {{ background: #222; padding: 10px; border-radius: 6px; margin-bottom: 8px; font-size: 13px; color: #ccc; word-break: break-all; }}
    h3 {{ margin-bottom: 10px; color: #aaa; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>🎬 YouTube Auto Uploader</h1>
  <div class="card">
    <h3>PASTE GOOGLE DRIVE LINK</h3>
    <form method="POST" action="/add">
      <input name="url" placeholder="https://drive.google.com/file/d/.../view" required />
      <input name="hint" placeholder="Short description e.g. ronaldo free kick goals" />
      <button type="submit">🚀 Add to Queue</button>
    </form>
    <div class="status">
      📋 In queue: <b>{queue_count}</b> &nbsp;|&nbsp;
      ✅ Posted: <b>{done_count}</b> &nbsp;|&nbsp;
      ⏱ Next auto post in: <b>{next_post}</b>
    </div>
    <div class="queue">
      <h3>CURRENT QUEUE</h3>
      {queue_items}
    </div>
  </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with lock:
            q = list(pending_links)
        done_count = 0
        if os.path.exists(DONE_FILE):
            with open(DONE_FILE) as f:
                done_count = len([l for l in f if l.strip()])

        # Calculate time until next auto post
        elapsed = time.time() - last_post_time[0]
        remaining = max(0, WAIT_SECONDS - elapsed)
        hrs  = int(remaining // 3600)
        mins = int((remaining % 3600) // 60)
        next_post = f"{hrs}h {mins}m" if remaining > 0 else "soon"

        items = "".join(
            f'<div class="item">🔗 {x["url"]}<br><small>{x.get("hint","")}</small></div>'
            for x in q
        ) or "<div class='item'>Empty — add a video or wait for auto post!</div>"

        html = HTML.format(queue_count=len(q), done_count=done_count, next_post=next_post, queue_items=items)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length).decode()
        params = parse_qs(body)
        url    = params.get("url", [""])[0].strip()
        hint   = params.get("hint", [""])[0].strip()
        if url:
            with lock:
                pending_links.append({"url": url, "hint": hint})
            print(f"  ➕  Added to queue: {url}")
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, *args):
        pass


def start_server():
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# ── Groq metadata ─────────────────────────────────────────────────────────────
def generate_metadata(hint):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return {"title": hint or "Amazing Video", "description": "Check this out!", "hashtags": "#viral #trending"}

    prompt = f"""Generate YouTube metadata for a video about: "{hint}"
Return ONLY valid JSON, no extra text:
{{
  "title": "catchy title under 70 chars, no hashtags",
  "description": "engaging 2-3 sentence description",
  "hashtags": "#tag1 #tag2 #tag3 #tag4 #tag5 #tag6"
}}"""

    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama3-8b-8192", "messages": [{"role": "user", "content": prompt}], "temperature": 0.9},
            timeout=15
        )
        text = res.json()["choices"][0]["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        print(f"  🤖  Title: {data['title']}")
        return data
    except Exception as e:
        print(f"  ⚠️  Groq failed: {e}")
        return {"title": hint or "Amazing Video", "description": "Check this out!", "hashtags": "#viral #trending #edit"}


# ── YouTube ───────────────────────────────────────────────────────────────────
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
    print("✅  Authenticated with YouTube")
    return build("youtube", "v3", credentials=creds)


def download_from_drive(url):
    out = os.path.join(DOWNLOAD_DIR, "video.mp4")
    if os.path.exists(out):
        os.remove(out)
    try:
        result = gdown.download(url, out, quiet=False, fuzzy=True)
        return result if result and os.path.exists(result) else None
    except Exception as e:
        print(f"  ❌  Download failed: {e}")
        return None


def upload_video(youtube, file_path, title, description, is_short=False):
    final_title = f"{title} #Shorts" if is_short else title
    body = {
        "snippet": {"title": final_title[:100], "description": description, "categoryId": "24"},
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
    try:
        req = youtube.videos().insert(part=",".join(body.keys()), body=body, media_body=media)
        response = None
        while response is None:
            status, response = req.next_chunk()
            if status:
                print(f"    ⬆️  {int(status.progress()*100)}%", end="\r")
        vid = response.get("id")
        kind = "Short" if is_short else "Video"
        print(f"    ✅  {kind} → https://youtube.com/watch?v={vid}    ")
        return vid
    except Exception as e:
        print(f"  ❌  Upload failed: {e}")
        return None


def mark_done(url):
    with open(DONE_FILE, "a") as f:
        f.write(url + "\n")


def process_video(youtube, item):
    url  = item["url"]
    hint = item.get("hint", "viral video")

    print(f"\n📥  Processing: {url}")
    print(f"  ⬇️  Downloading...")

    file_path = download_from_drive(url)
    if not file_path:
        print("  ❌  Download failed. Skipping.")
        return

    size_mb = os.path.getsize(file_path) / (1024*1024)
    print(f"  📦  {size_mb:.1f} MB")

    meta        = generate_metadata(hint)
    title       = meta["title"]
    description = f"{meta['description']}\n\n{meta['hashtags']}"

    print(f"  📹  Posting as Video...")
    upload_video(youtube, file_path, title, description, is_short=False)
    time.sleep(5)
    print(f"  🎬  Posting as Short...")
    upload_video(youtube, file_path, title, description, is_short=True)

    try:
        os.remove(file_path)
    except OSError:
        pass

    mark_done(url)
    last_post_time[0] = time.time()
    print(f"  ✅  Done!\n")


# ── Bot loop ──────────────────────────────────────────────────────────────────
def bot_loop():
    youtube = get_youtube_client()
    last_post_time[0] = time.time()
    print("🤖  Bot running. Dashboard + auto post every 4 hours.\n")

    while True:
        with lock:
            item = pending_links.pop(0) if pending_links else None

        if item:
            # Someone submitted via dashboard — post it now
            process_video(youtube, item)
        else:
            # Check if 4 hours have passed since last post
            elapsed = time.time() - last_post_time[0]
            if elapsed >= WAIT_SECONDS:
                print("⏰  4 hours passed, auto posting default video...")
                process_video(youtube, DEFAULT_VIDEO)
            else:
                time.sleep(10)


def main():
    threading.Thread(target=start_server, daemon=True).start()
    bot_loop()


if __name__ == "__main__":
    main()
