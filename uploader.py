"""
YouTube Auto Uploader — Render Edition
----------------------------------------
Reads videos.csv, downloads each pending video, uploads to YouTube,
then marks it as done in a done.txt log so it never re-uploads.

Environment variables (already set in Render):
  YOUTUBE_CLIENT_ID      → your OAuth client ID
  YOUTUBE_CLIENT_SECRET  → your OAuth client secret
  YOUTUBE_REFRESH_TOKEN  → your refresh token (generated once, see README)

Run manually or as a Render Cron Job:
  python uploader.py
"""

import os
import sys
import csv
import time
import tempfile

import yt_dlp
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ── Config ───────────────────────────────────────────────────────────────────
CSV_FILE     = "videos.csv"
DONE_FILE    = "done.txt"
DOWNLOAD_DIR = tempfile.gettempdir()
# ─────────────────────────────────────────────────────────────────────────────


def get_youtube_client():
    """Build YouTube API client using env var credentials (no browser needed)."""
    client_id     = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")

    if not all([client_id, client_secret, refresh_token]):
        print("❌  Missing one or more env vars:")
        print("    YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN")
        sys.exit(1)

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )

    creds.refresh(Request())
    print("✅  Authenticated with YouTube API\n")
    return build("youtube", "v3", credentials=creds)


def load_done() -> set:
    if not os.path.exists(DONE_FILE):
        return set()
    with open(DONE_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_done(url: str):
    with open(DONE_FILE, "a") as f:
        f.write(url + "\n")


def download_video(url: str) -> str | None:
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
            if not path.endswith(".mp4"):
                path = os.path.splitext(path)[0] + ".mp4"
            return path
    except Exception as e:
        print(f"  ❌  Download failed: {e}")
        return None


def upload_video(youtube, file_path, title, description, category_id, privacy):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)
    try:
        request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media,
        )
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"    ⬆️  Uploading... {int(status.progress() * 100)}%", end="\r")
        video_id = response.get("id")
        print(f"    ✅  Done → https://youtube.com/watch?v={video_id}    ")
        return video_id
    except Exception as e:
        print(f"  ❌  Upload failed: {e}")
        return None


def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌  {CSV_FILE} not found.")
        sys.exit(1)

    youtube = get_youtube_client()
    done    = load_done()

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pending = [r for r in rows if r["url"].strip() not in done]
    print(f"📋  {len(pending)} pending video(s) out of {len(rows)} total.\n")

    if not pending:
        print("✅  Nothing new to upload. Add more rows to videos.csv.")
        return

    for i, row in enumerate(pending, 1):
        url         = row["url"].strip()
        title       = row.get("title", "Untitled").strip()
        description = row.get("notes", "").strip()
        category    = row.get("category_id", "24").strip()
        privacy     = row.get("privacy", "public").strip().lower()

        print(f"[{i}/{len(pending)}] {title}")
        print(f"  🔗  {url}")
        print("  ⬇️  Downloading...")

        file_path = download_video(url)
        if not file_path or not os.path.exists(file_path):
            print("  ⚠️  Skipping.\n")
            continue

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  📦  {size_mb:.1f} MB downloaded")

        video_id = upload_video(youtube, file_path, title, description, category, privacy)

        try:
            os.remove(file_path)
        except OSError:
            pass

        if video_id:
            mark_done(url)

        print()
        if i < len(pending):
            time.sleep(3)

    print("🎉  All done!")


if __name__ == "__main__":
    main()
