"""
YouTube Auto Uploader — Runs forever
- Every hour, posts ALL videos in videos.csv one by one
- Never skips, never exits
- Add new videos to CSV anytime — they get posted next round
"""

import os
import csv
import time
import tempfile

import gdown
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

CSV_FILE     = "videos.csv"
DOWNLOAD_DIR = tempfile.gettempdir()
WAIT_SECONDS = 3600  # 1 hour between each video


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


def main():
    youtube = get_youtube_client()
    print("🤖  Bot is running. Posts 1 video per hour forever.\n")

    while True:
        videos = load_videos()

        if not videos:
            print("😴  No videos in CSV. Add some and I'll pick them up next hour.")
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


if __name__ == "__main__":
    main()
