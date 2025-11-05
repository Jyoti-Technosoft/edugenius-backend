import os, io, json, base64, datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


# -----------------------------------------------------------
# .ENV LOADER WITH DEBUG LOGGING
# -----------------------------------------------------------


def load_env():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v
    else:
        print("⚠️ .env file not found.")


load_env()

SCOPES = [os.environ.get("SCOPES")]
FOLDER_ID = os.environ.get("FOLDER_ID")


def decode_json_env_var(key):
    """Decode base64 JSON data from .env"""
    data = os.environ.get(key)
    if not data:
        return None

    try:
        decoded = base64.b64decode(data).decode()
        parsed = json.loads(decoded)

        return parsed
    except Exception as e:
        print(f"❌ Failed to decode/parse {key}: {str(e)}")
        return None


def save_token_to_env(creds):
    """Save refreshed token back to .env as base64"""
    try:
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

    except Exception as e:
        print(f"❌ Failed to save token to .env: {str(e)}")

# -----------------------------------------------------------
# GOOGLE AUTH USING BASE64 JSON
# -----------------------------------------------------------
def get_drive_service():
    creds = None

    # -----------------------------------------------------------
    # STEP 1: Print raw .env values BEFORE decoding
    # -----------------------------------------------------------
    raw_client_secret = os.environ.get("CLIENT_SECRET")
    raw_token = os.environ.get("TOKEN")



    if raw_client_secret:
        print(f"   CLIENT_SECRET (preview): {raw_client_secret[:60]}...")
    if raw_token:
        print(f"   TOKEN (preview): {raw_token[:60]}...")

    # -----------------------------------------------------------
    # STEP 2: Decode CLIENT_SECRET
    # -----------------------------------------------------------

    client_secret = decode_json_env_var("CLIENT_SECRET")

    # -----------------------------------------------------------
    # STEP 3: Decode TOKEN
    # -----------------------------------------------------------
    token_data = decode_json_env_var("TOKEN")

    # -----------------------------------------------------------
    # STEP 4: Load credentials
    # -----------------------------------------------------------

    if token_data:
        try:
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)
            print("✅ Loaded credentials from TOKEN env.")
        except Exception as e:
            print(f"❌ Error loading credentials from TOKEN: {str(e)}")

    # -----------------------------------------------------------
    # STEP 5: Refresh or start login if needed
    # -----------------------------------------------------------
    if creds and creds.expired and creds.refresh_token:
        print("🔁 Refreshing expired token...")
        try:
            creds.refresh(Request())
            save_token_to_env(creds)
        except Exception as e:
            print(f"❌ Failed to refresh token: {str(e)}")

    if not creds or not creds.valid:
        print("⚠️ No valid token found, starting console login...")
        try:
            flow = InstalledAppFlow.from_client_config(client_secret, SCOPES)
            creds = flow.run_console()
            save_token_to_env(creds)
        except Exception as e:
            print(f"❌ Failed during InstalledAppFlow: {str(e)}")
            raise

    # -----------------------------------------------------------
    # STEP 6: Initialize Drive service
    # -----------------------------------------------------------
    print("\n✅ Google Drive service initialized successfully.")
    return build("drive", "v3", credentials=creds)
# -----------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------
def upload_to_drive(file_path):
    try:
        service = get_drive_service()
        file_metadata = {'name': os.path.basename(file_path), 'parents': [FOLDER_ID]}
        media = MediaFileUpload(file_path, resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        file_id = file.get('id')
        print(f"Uploaded {file_path} → Drive (File ID: {file_id})")
        return file_id
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        raise


# -----------------------------------------------------------
# IN-MEMORY DOWNLOAD
# -----------------------------------------------------------
def get_file_content_in_memory(file_id):
    try:
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
        print(f"✅ File {file_id} downloaded into memory successfully.")
        print("File loaded into memory successfully.")
        return file_stream
    except Exception as e:
        print(f"❌ File download failed for {file_id}: {str(e)}")
        raise


def delete_from_drive(file_id):
    try:
        service = get_drive_service()
        service.files().delete(fileId=file_id).execute()
        print(f"🗑️ Deleted file from Drive (File ID: {file_id})")
        return True
    except Exception as e:
        print(f"❌ Failed to delete file {file_id} from Drive: {str(e)}")
        return False
