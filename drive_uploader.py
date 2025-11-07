import os, io, json, base64, datetime


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
