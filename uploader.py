"""
YouTube Auto Uploader — Render Edition
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

CSV_FILE     = "videos.csv"
DONE_FILE    = "done.txt"
DOWNLOAD_DIR = tempfile.gettempdir()
COOKIES_FILE = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")


def write_cookies_file():
    cookies = os.environ.get("YOUTUBE_COOKIES", "").strip()
    if not cookies:
        print("⚠️  YOUTUBE_COOKIES not set")
        return None
    with open(COOKIES_FILE, "w") as f:
        f.write(cookies)
    print("✅  Cookies loaded")
    return COOKIES_FILE


def get_youtube_client():
    client_id     = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")

    if not all([client_id, client_secret, refresh_token]):
        print("❌  Missing YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET / YOUTUBE_REFRESH_TOKEN")
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
    print("✅  Authenticated with YouTube API")
    return build("youtube", "v3", credentials=creds)


def load_done():
    if not os.path.exists(DONE_FILE):
        return set()
    with open(DONE_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_done(url):
    with open(DONE_FILE, "a") as f:
        f.write(url + "\n")


def download_video(url, cookies_file):
    """Try multiple format strategies until one works."""
    formats_to_try = [
        "best[ext=mp4]",
        "best",
        "worst",
    ]

    for fmt in formats_to_try:
        ydl_opts = {
            "format": fmt,
            "outtmpl": os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        if cookies_file and os.path.exists(cookies_file):
            ydl_opts["cookiefile"] = cookies_file

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Find the actual downloaded file
                video_id = info.get("id", "")
                for f in os.listdir(DOWNLOAD_DIR):
                    if video_id in f:
                        full_path = os.path.join(DOWNLOAD_DIR, f)
                        # Rename to .mp4 if needed
                        if not full_path.endswith(".mp4"):
                            new_path = os.path.splitext(full_path)[0] + ".mp4"
                            os.rename(full_path, new_path)
                            full_path = new_path
                        return full_path
        except Exception as e:
            print(f"  ⚠️  Format '{fmt}' failed: {e}")
            continue

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
        print(f"    ✅  Uploaded → https://youtube.com/watch?v={video_id}    ")
        return video_id
    except Exception as e:
        print(f"  ❌  Upload failed: {e}")
        return None


def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌  {CSV_FILE} not found.")
        sys.exit(1)

    cookies_file = write_cookies_file()
    youtube      = get_youtube_client()
    done         = load_done()

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pending = [r for r in rows if r["url"].strip() not in done]
    print(f"\n📋  {len(pending)} pending / {len(rows)} total\n")

    if not pending:
        print("✅  Nothing new to upload.")
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

        file_path = download_video(url, cookies_file)
        if not file_path or not os.path.exists(file_path):
            print("  ❌  All download attempts failed. Skipping.\n")
            continue

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  📦  {size_mb:.1f} MB — uploading...")

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
