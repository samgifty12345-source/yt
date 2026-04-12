"""
Run this ONCE on your local machine to get your YOUTUBE_REFRESH_TOKEN.
It will open a browser, ask you to log in, then print the token.

Steps:
  1. pip install google-auth-oauthlib
  2. Put your client_secrets.json in this folder
  3. python get_refresh_token.py
  4. Copy the printed refresh token into Render env vars
"""

import json
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
creds = flow.run_local_server(port=0)

print("\n" + "="*60)
print("✅  Copy these into your Render environment variables:")
print("="*60)
print(f"\nYOUTUBE_CLIENT_ID     = {creds.client_id}")
print(f"YOUTUBE_CLIENT_SECRET = {creds.client_secret}")
print(f"YOUTUBE_REFRESH_TOKEN = {creds.refresh_token}")
print("\n" + "="*60)
