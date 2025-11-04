from __future__ import print_function
import os, io, json, base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


# -----------------------------------------------------------
# .ENV LOADER
# -----------------------------------------------------------
def load_env():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v

load_env()

SCOPES = [os.environ.get("SCOPES")]
FOLDER_ID = os.environ.get("FOLDER_ID")


def decode_json_env_var(key):
    data = os.environ.get(key)
    if not data:
        return None
    return json.loads(base64.b64decode(data).decode())


def save_token_to_env(creds):
    token_json = creds.to_json()
    encoded = base64.b64encode(token_json.encode()).decode()

    lines = []
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if not line.startswith("TOKEN="):
                    lines.append(line)

    lines.append(f"TOKEN={encoded}\n")

    with open(".env", "w") as f:
        f.writelines(lines)

    print("✅ Token updated in .env")


# -----------------------------------------------------------
# GOOGLE AUTH USING BASE64 JSON
# -----------------------------------------------------------
def get_drive_service():
    creds = None

    client_secret = decode_json_env_var("CLIENT_SECRET")
    token_data = decode_json_env_var("TOKEN")

    # Load token from .env
    if token_data:
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)

    # Refresh OR login first time
    if creds and creds.expired and creds.refresh_token:
        print("🔁 Refreshing expired token...")
        creds.refresh(Request())
        save_token_to_env(creds)

    if not creds or not creds.valid:
        print("⚠️ No valid token found, starting console login...")
        flow = InstalledAppFlow.from_client_config(client_secret, SCOPES)
        creds = flow.run_console()
        save_token_to_env(creds)

    return build("drive", "v3", credentials=creds)


# -----------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------
def upload_to_drive(file_path):
    service = get_drive_service()
    file_metadata = {'name': os.path.basename(file_path), 'parents': [FOLDER_ID]}
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    print(f"Uploaded {file_path} → Drive (File ID: {file_id})")
    return file_id


# -----------------------------------------------------------
# IN-MEMORY DOWNLOAD
# -----------------------------------------------------------
def get_file_content_in_memory(file_id):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"Download progress: {int(status.progress() * 100)}%")

    file_stream.seek(0)
    print("File loaded into memory successfully.")
    return file_stream
