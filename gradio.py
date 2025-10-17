

# from gradio_client import Client
# client = Client("heerjtdev/edugenius")
#
# # This will print a summary of all available API endpoints (functions).
# client.view_api()


import json
import os
from gradio_client import Client
from typing import Dict, Any, Tuple


# Define a custom ConnectionError or use a built-in one if needed.
# Since we are using standard exceptions, we'll ensure we handle common ones.
# Note: gradio_client will typically raise a ValueError or similar on connection failure.

def process_pdf_pipeline(pdf_path: str) -> Dict[str, Any]:
    """
    Calls the heerjtdev/edugenius Hugging Face Space API to process a PDF.

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
        client = Client("heerjtdev/edugenius")
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


