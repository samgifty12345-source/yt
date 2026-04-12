"""
YouTube Auto Uploader — Google Drive Edition
Downloads videos from Google Drive links in videos.csv and uploads to YouTube.
"""

import os
import sys
import csv
import time
import tempfile

import gdown
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

CSV_FILE     = "videos.csv"
DONE_FILE    = "done.txt"
DOWNLOAD_DIR = tempfile.gettempdir()


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


def load_done():
    if not os.path.exists(DONE_FILE):
        return set()
    with open(DONE_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def mark_done(url):
    with open(DONE_FILE, "a") as f:
        f.write(url + "\n")


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
    done    = load_done()

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pending = [r for r in rows if r["drive_url"].strip() not in done]
    print(f"\n📋  {len(pending)} pending / {len(rows)} total\n")

    if not pending:
        print("✅  Nothing new. Add rows to videos.csv.")
        return

    for i, row in enumerate(pending, 1):
        url         = row["drive_url"].strip()
        title       = row.get("title", "Untitled").strip()
        description = row.get("notes", "").strip()
        category    = row.get("category_id", "24").strip()
        privacy     = row.get("privacy", "public").strip().lower()

        print(f"[{i}/{len(pending)}] {title}")
        print(f"  ⬇️  Downloading from Drive...")

        file_path = download_from_drive(url)
        if not file_path:
            print("  ❌  Skipping.\n")
            continue

        size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"  📦  {size_mb:.1f} MB")

        vid = upload_video(youtube, file_path, title, description, category, privacy)

        try:
            os.remove(file_path)
        except OSError:
            pass

        if vid:
            mark_done(url)

        print()
        if i < len(pending):
            time.sleep(3)

    print("🎉  Done!")


if __name__ == "__main__":
    main()
