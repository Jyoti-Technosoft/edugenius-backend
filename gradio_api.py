
# # This will print a summary of all available API endpoints (functions).
# client.view_api()


import os,tempfile
import json
from typing import Dict, Any, Tuple
from gradio_client import Client
from drive_uploader import load_env



# Define a custom ConnectionError or use a built-in one if needed.
# Since we are using standard exceptions, we'll ensure we handle common ones.
# Note: gradio_client will typically raise a ValueError or similar on connection failure.

def call_edugenius_api(pdf_path: str) -> Dict[str, Any]:
    load_env()  # make sure env is loaded before using
    hf_space = os.environ.get("HF_SPACE_E")
    if not hf_space:
        raise RuntimeError("HF_SPACE not found in .env")
    """

    Args:
        pdf_path: The local file path to the PDF document.

    Returns:
        A dictionary (JSON object) containing the structured MCQ output.

    Raises:
        ValueError: If the API call fails due to invalid file or processing error.
    """
    # 1. Check if the local file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Local PDF file not found at: {pdf_path}")

    # 2. Initialize the client with the Space's ID
    try:
        # Client initialization is non-blocking and fast
        client = Client(hf_space)
    except Exception as e:
        # This handles issues like incorrect Space ID or network problems during setup
        raise ConnectionError(f"Could not initialize connection to Hugging Face Space: {e}")

    # 3. Call the prediction function using the confirmed api_name
    print(f"Uploading and processing file: {pdf_path}...")

    # FIX: Manually create the structured input dictionary (FileData)
    # to satisfy the Gradio API's strict validation requirements.
    structured_file_input = {
        "path": pdf_path,
        "meta": {"_type": "gradio.FileData"}
    }

    try:
        # The client.predict returns the two outputs as a tuple: (status_message, structured_mcq_json_output)
        response: Tuple[str, str | dict | list] = client.predict(
            structured_file_input,  # Pass the structured dictionary here
            api_name="/predict"  # The confirmed endpoint
        )
    except Exception as e:
        raise ValueError(f"API call failed during prediction phase: {e}")

    status_message = response[0]
    structured_mcq_output = response[1]  # This is the main JSON output

    # 4. Handle potential errors and parse the output
    if "Error" in status_message:
        raise ValueError(f"API processing error reported by Space: {status_message}")

    # If the output component in Gradio is 'Json', it often returns a Python dict/list directly.
    if isinstance(structured_mcq_output, (dict, list)):
        return structured_mcq_output

    # If it was returned as a string (e.g., from a Textbox component), parse it.
    try:
        return json.loads(structured_mcq_output)
    except (json.JSONDecodeError, TypeError):
        # Fallback for unexpected output format
        print(f"Warning: Final output was not standard JSON. Status: {status_message}")
        return {"status_message": status_message, "raw_output": structured_mcq_output}


def call_layoutlm_api(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Send an in-memory PDF or image directly to the Hugging Face model (no Drive upload).
    Supports .pdf, .png, .jpg, .jpeg automatically.
    """
    import mimetypes

    load_env()
    hf_space = os.environ.get("HF_SPACE")
    hf_token = os.environ.get("HF_SPACE_TOKEN")
    if not hf_space:
        raise RuntimeError("HF_SPACE not found in .env")

    print(f"[INFO] Connecting to Hugging Face Space: {hf_space}")
    try:
        client = Client(hf_space,hf_token=hf_token)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Hugging Face Space: {e}")

    # ✅ detect correct extension
    ext = os.path.splitext(filename)[-1].lower()

    # ✅ create temp file with same extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    print(f"[STEP] Sending {filename} ({ext}) to LayoutLM model...")

    structured_input_list = [{
        "path": tmp_path,
        "meta": {"_type": "gradio.FileData"}
    }]

    try:
        response = client.predict(structured_input_list, api_name="/predict")
    except Exception as e:
        raise ValueError(f"LayoutLM API call failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

    # ✅ Normalize response
    if isinstance(response, (dict, list)):
        return response

    if isinstance(response, tuple):
        for item in response:
            if isinstance(item, (dict, list)):
                return item
            if isinstance(item, str):
                try:
                    return json.loads(item)
                except json.JSONDecodeError:
                    continue
        return {"raw_output": response}

    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_output": response}

    return {"raw_output": response}



def call_yolo_api(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Send an in-memory PDF or image directly to the Hugging Face model (no Drive upload).
    Supports .pdf, .png, .jpg, .jpeg automatically.
    """
    import mimetypes

    load_env()
    hf_space = os.environ.get("HF_SPACE")
    hf_token = os.environ.get("HF_SPACE_TOKEN")

    if not hf_space:
        raise RuntimeError("HF_SPACE not found in .env")

    print(f"[INFO] Connecting to Hugging Face Space: {hf_space}")
    try:
        client = Client(hf_space,hf_token=hf_token)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Hugging Face Space: {e}")

    # ✅ detect correct extension
    ext = os.path.splitext(filename)[-1].lower()

    # ✅ create temp file with same extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    print(f"[STEP] Sending {filename} ({ext}) to LayoutLM model...")

    structured_input_list = {
        "_type": "gradio.FileData",
        "path": tmp_path,
        "meta": {
            "_type": "gradio.FileData"
        }
    }

    try:
        response = client.predict(structured_input_list, api_name="/process_document")
    except Exception as e:
        raise ValueError(f"LayoutLM API call failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

    # ✅ Normalize response
    if isinstance(response, (dict, list)):
        return response

    if isinstance(response, tuple):
        for item in response:
            if isinstance(item, (dict, list)):
                return item
            if isinstance(item, str):
                try:
                    return json.loads(item)
                except json.JSONDecodeError:
                    continue
        return {"raw_output": response}

    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_output": response}

    return {"raw_output": response}

def latex_model(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Send an in-memory PDF or image directly to the Hugging Face model (no Drive upload).
    Supports .pdf, .png, .jpg, .jpeg automatically.
    """
    import mimetypes

    load_env()
    hf_space = os.environ.get("HF_SPACE_LATEX")

    if not hf_space:
        raise RuntimeError("HF_SPACE not found in .env")

    print(f"[INFO] Connecting to Hugging Face Space: {hf_space}")
    try:
        client = Client(hf_space)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Hugging Face Space: {e}")

    # ✅ detect correct extension
    ext = os.path.splitext(filename)[-1].lower()

    # ✅ create temp file with same extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    print(f"[STEP] Sending {filename} ({ext}) to LayoutLM model...")

    structured_input_list = {
        "_type": "gradio.FileData",
        "path": tmp_path,
        "meta": {
            "_type": "gradio.FileData"
        }
    }

    try:
        response = client.predict(structured_input_list, api_name="/process_document")
    except Exception as e:
        raise ValueError(f"LayoutLM API call failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

    # ✅ Normalize response
    if isinstance(response, (dict, list)):
        return response

    if isinstance(response, tuple):
        for item in response:
            if isinstance(item, (dict, list)):
                return item
            if isinstance(item, str):
                try:
                    return json.loads(item)
                except json.JSONDecodeError:
                    continue
        return {"raw_output": response}

    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_output": response}

    return {"raw_output": response}
