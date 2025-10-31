from __future__ import print_function
import os
import io
import zipfile
from typing import List
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
CHROMA_PATH = "./chromadb_data"       # your vector db local folder
ZIP_FILE = "chroma_backup.zip"
CHUNKS_DIR = "chroma_chunks"
CHUNK_SIZE_MB = 20                    # max chunk size (Drive-friendly)
RESTORED_ZIP = "restored_backup.zip"

SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = '1CKTqw6-CBBDd_OoJjKmv5NoUStjn4q0z'  # replace with your actual Drive folder ID

# -----------------------------------------------------------
# DRIVE AUTH
# -----------------------------------------------------------
def get_drive_service():
    """Authenticate and return an authorized Drive service."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret_1.json', SCOPES)
            creds = flow.run_console(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

# -----------------------------------------------------------
# ZIP + CHUNK OPERATIONS
# -----------------------------------------------------------
def zip_chroma_folder():
    """Compress the Chroma vector DB folder into a zip file."""
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError("Chroma folder not found.")
    print("Zipping Chroma data...")
    with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(CHROMA_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, CHROMA_PATH)
                zipf.write(file_path, arcname)
    print(f"Created {ZIP_FILE}")

def split_zip_into_chunks():
    """Split the zip into smaller chunks for upload."""
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    chunk_size = CHUNK_SIZE_MB * 1024 * 1024
    with open(ZIP_FILE, "rb") as f:
        i = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            with open(f"{CHUNKS_DIR}/chunk_{i}.part", "wb") as chunk_file:
                chunk_file.write(chunk)
            i += 1
    print(f"Split into {i} chunks")

# -----------------------------------------------------------
# DRIVE UPLOAD / DOWNLOAD
# -----------------------------------------------------------
def upload_to_drive(file_path):
    """Uploads a file to a specific Drive folder and returns its ID."""
    service = get_drive_service()
    file_metadata = {'name': os.path.basename(file_path), 'parents': [FOLDER_ID]}
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    print(f"Uploaded {file_path} → Drive (File ID: {file_id})")
    return file_id

def upload_chunks_to_drive() -> List[str]:
    """Uploads all chunks in CHUNKS_DIR to Drive folder."""
    service = get_drive_service()
    uploaded_ids = []
    for filename in sorted(os.listdir(CHUNKS_DIR)):
        file_path = os.path.join(CHUNKS_DIR, filename)
        file_metadata = {'name': filename, 'parents': [FOLDER_ID]}
        media = MediaFileUpload(file_path, resumable=True)
        uploaded = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        file_id = uploaded.get("id")
        uploaded_ids.append(file_id)
        print(f"Uploaded {filename} -> {file_id}")
    return uploaded_ids

def get_file_url(file_id):
    """Generate a public view link for a file."""
    return f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"

def download_and_rebuild_zip(file_ids: List[str]):
    """Download chunks from Drive and rebuild into one zip file."""
    service = get_drive_service()
    os.makedirs("downloaded_chunks", exist_ok=True)
    for i, file_id in enumerate(file_ids):
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(f"downloaded_chunks/chunk_{i}.part", "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Downloading chunk {i} - {int(status.progress() * 100)}%")
    # Combine parts
    with open(RESTORED_ZIP, "wb") as out:
        for i in range(len(file_ids)):
            with open(f"downloaded_chunks/chunk_{i}.part", "rb") as f:
                out.write(f.read())
    print("Rebuilt restored_backup.zip")

def unzip_restored_backup():
    """Extracts the restored zip file into the Chroma folder."""
    with zipfile.ZipFile(RESTORED_ZIP, "r") as zip_ref:
        zip_ref.extractall(CHROMA_PATH)
    print("Unzipped into chromadb_data folder.")

# -----------------------------------------------------------
# IN-MEMORY DOWNLOAD
# -----------------------------------------------------------
def get_file_content_in_memory(file_id):
    """Fetches file from Drive and returns a BytesIO stream."""
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

# -----------------------------------------------------------
# PIPELINE SHORTCUTS
# -----------------------------------------------------------
def backup_chroma_to_drive():
    """Full backup: zip -> chunk -> upload -> return file IDs."""
    zip_chroma_folder()
    split_zip_into_chunks()
    ids = upload_chunks_to_drive()
    print("Uploaded all chunks to Drive.")
    return ids

def restore_chroma_from_drive(file_ids: List[str]):
    """Full restore: download -> rebuild -> unzip."""
    download_and_rebuild_zip(file_ids)
    unzip_restored_backup()

    print("Chroma DB restored successfully.")
