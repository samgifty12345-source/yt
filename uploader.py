"""
YouTube Auto Uploader — Runs forever with dummy web server
"""

import os
import csv
import time
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import gdown
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

CSV_FILE     = "videos.csv"
DOWNLOAD_DIR = tempfile.gettempdir()
WAIT_SECONDS = 3600


# Dummy server to keep Render happy
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running!")
    def log_message(self, *args):
        pass  # silence logs

def start_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


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


def load_videos():
    if not os.path.exists(CSV_FILE):
        return []
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("drive_url", "").strip()]


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


def upload_video(youtube, file_path, title, description, category_id, privacy):
    body = {
        "snippet": {"title": title, "description": description, "categoryId": category_id},
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
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
        print(f"    ✅  https://youtube.com/watch?v={vid}    ")
        return vid
    except Exception as e:
        print(f"  ❌  Upload failed: {e}")
        return None


def bot_loop():
    youtube = get_youtube_client()
    print("🤖  Bot is running. Posts 1 video per hour forever.\n")

    while True:
        videos = load_videos()

        if not videos:
            print("😴  No videos in CSV. Checking again in 1 hour...")
            time.sleep(WAIT_SECONDS)
            continue

        print(f"📋  {len(videos)} video(s) in CSV\n")

        for i, row in enumerate(videos, 1):
            url         = row["drive_url"].strip()
            title       = row.get("title", "Untitled").strip()
            description = row.get("notes", "").strip()
            category    = row.get("category_id", "24").strip()
            privacy     = row.get("privacy", "public").strip().lower()

            print(f"[{i}/{len(videos)}] {title}")
            print(f"  ⬇️  Downloading...")

            file_path = download_from_drive(url)
            if not file_path:
                print("  ❌  Skipping.\n")
                time.sleep(WAIT_SECONDS)
                continue

            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"  📦  {size_mb:.1f} MB")

            upload_video(youtube, file_path, title, description, category, privacy)

            try:
                os.remove(file_path)
            except OSError:
                pass

            print(f"\n⏳  Waiting 1 hour before next video...\n")
            time.sleep(WAIT_SECONDS)


def main():
    # Start dummy web server in background thread
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # Run bot in main thread
    bot_loop()


if __name__ == "__main__":
    main()
