
# # This will print a summary of all available API endpoints (functions).
# client.view_api()


import os,tempfile
import json
from typing import Dict, Any, Tuple
from gradio_client import Client
from drive_uploader import load_env
import time



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

    # load_env()
    # hf_space = os.environ.get("HF_SPACE")
    hf_space = 'heerjtdev/layout_latex'
    # hf_token = os.environ.get("HF_SPACE_TOKEN")

    if not hf_space:
        raise RuntimeError("HF_SPACE not found in .env")

    print(f"[INFO] Connecting to Hugging Face Space: {hf_space}")
    try:
        # client = Client(hf_space,hf_token=hf_token)
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

def latex_model(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Send an in-memory PDF or image directly to the Hugging Face model (no Drive upload).
    Supports .pdf, .png, .jpg, .jpeg automatically.
    """
    import mimetypes

    load_env()
    hf_space = "heerjtdev/layout_latex"

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
        response = client.predict([structured_input_list],"", api_name="/process_file")
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






#
# def call_feeedback_api(file_bytes: bytes, filename: str) -> Dict[str, Any]:
#     """
#     Calls the heerjtdev/feeedback Hugging Face Space to analyze a PDF.
#     It includes detailed time logs for debugging latency issues.
#     """
#     # Record start time for total execution
#     start_time = time.time()
#
#     # load_env() # Uncomment if needed
#     hf_space = "heerjtdev/feeedback"  # Space URL
#
#     print(f"[INFO] Connecting to Hugging Face Space: {hf_space}")
#
#     # 1. Initialize the client
#     client_start = time.time()
#     try:
#         client = Client(hf_space)
#         client_duration = time.time() - client_start
#         print(f"[TIMELOG] Client Initialization took: {client_duration:.2f} seconds")
#     except Exception as e:
#         raise ConnectionError(f"Could not initialize connection to Hugging Face Space: {e}")
#
#     ext = os.path.splitext(filename)[-1].lower()
#     tmp_path = None
#
#     try:
#         # 2. Create temp file from in-memory bytes
#         temp_file_start = time.time()
#         with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
#             tmp.write(file_bytes)
#             tmp_path = tmp.name
#         temp_file_duration = time.time() - temp_file_start
#         print(f"[TIMELOG] Temp File Creation took: {temp_file_duration:.2f} seconds")
#
#         print(f"[STEP] Processing temporary file: {tmp_path}...")
#
#         # Prepare the structured input dictionary (FileData)
#         structured_file_input = {
#             "path": tmp_path,
#             "meta": {"_type": "gradio.FileData"}
#         }
#
#         # 3. Call the prediction function (THE SLOW STEP: Cold Start + Model Run)
#         api_call_start = time.time()
#         response: Tuple[Any, ...] = client.predict(
#             structured_file_input,
#             api_name="/gradio_process_pdf"
#         )
#         api_call_duration = time.time() - api_call_start
#         print(f"[TIMELOG] API Prediction Call took: {api_call_duration:.2f} seconds")
#
#         # 4. Parse the 5-item response (List/Tuple)
#         if isinstance(response, (list, tuple)) and len(response) >= 3:
#             num_pages_str = str(response[0])
#             num_equations_str = str(response[1])
#             num_figures_str = str(response[2])
#
#             # Map the results to the keys expected by your Flask route, converting to int
#             final_result = {
#                 "Total Pages in PDF": int(num_pages_str) if num_pages_str.isdigit() else num_pages_str,
#                 "Total Equations Detected": int(
#                     num_equations_str) if num_equations_str.isdigit() else num_equations_str,
#                 "Total Figures Detected": int(num_figures_str) if num_figures_str.isdigit() else num_figures_str,
#             }
#
#             total_duration = time.time() - start_time
#             print(f"[TIMELOG] Total function execution took: {total_duration:.2f} seconds")
#             return final_result
#
#         # Fallback for unexpected format
#         total_duration = time.time() - start_time
#         print(f"[TIMELOG] Total function execution took (failed): {total_duration:.2f} seconds")
#         return {"raw_output": response, "error": "Unexpected output format from API. Expected a 5-item list."}
#
#     except Exception as e:
#         # Re-raise exceptions as ValueError for API failure
#         total_duration = time.time() - start_time
#         print(f"[TIMELOG] Total function execution took (exception): {total_duration:.2f} seconds")
#         raise ValueError(f"API call failed during prediction phase: {e}")
#
#     finally:
#         # 5. Clean up the temporary file
#         if tmp_path and os.path.exists(tmp_path):
#             try:
#                 os.remove(tmp_path)
#             except Exception as e:
#                 print(f"[WARN] Failed to delete temporary file {tmp_path}: {e}")


def call_feeedback_api(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Calls the heerjtdev/feeedback Hugging Face Space to analyze a PDF.
    It now parses the six-item response, including the per-page equation counts.
    """
    # Record start time for total execution
    start_time = time.time()

    # load_env() # Uncomment if needed
    hf_space = "heerjtdev/feeedback"  # Space URL

    print(f"[INFO] Connecting to Hugging Face Space: {hf_space}")

    # 1. Initialize the client
    client_start = time.time()
    try:
        client = Client(hf_space)
        client_duration = time.time() - client_start
        print(f"[TIMELOG] Client Initialization took: {client_duration:.2f} seconds")
    except Exception as e:
        raise ConnectionError(f"Could not initialize connection to Hugging Face Space: {e}")

    ext = os.path.splitext(filename)[-1].lower()
    tmp_path = None

    try:
        # 2. Create temp file from in-memory bytes
        temp_file_start = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        temp_file_duration = time.time() - temp_file_start
        print(f"[TIMELOG] Temp File Creation took: {temp_file_duration:.2f} seconds")

        print(f"[STEP] Processing temporary file: {tmp_path}...")

        # Prepare the structured input dictionary (FileData)
        structured_file_input = {
            "path": tmp_path,
            "meta": {"_type": "gradio.FileData"}
        }

        # 3. Call the prediction function (THE SLOW STEP: Cold Start + Model Run)
        api_call_start = time.time()
        # Expecting a 6-item tuple: (pages, eq_total, fig_total, report, page_counts_dict, gallery_list)
        response: Tuple[Any, ...] = client.predict(
            structured_file_input,
            api_name="/gradio_process_pdf"
        )
        api_call_duration = time.time() - api_call_start
        print(f"[TIMELOG] API Prediction Call took: {api_call_duration:.2f} seconds")

        # 4. Parse the 6-item response (List/Tuple)
        # We need at least 5 items (index 0 through 4) to get the page counts dict
        if isinstance(response, (list, tuple)) and len(response) >= 5:
            num_pages_str = str(response[0])
            num_equations_str = str(response[1])
            num_figures_str = str(response[2])

            # Extract the 5th item (index 4), which is the Dict[str, int] of page counts
            page_counts_dict = response[4]

            # Map the results to the keys expected by your Flask route
            final_result = {
                "Total Pages in PDF": int(num_pages_str) if num_pages_str.isdigit() else num_pages_str,
                "Total Equations Detected": int(
                    num_equations_str) if num_equations_str.isdigit() else num_equations_str,
                "Total Figures Detected": int(num_figures_str) if num_figures_str.isdigit() else num_figures_str,
                # NEW KEY ADDED
                "Equation Counts Per Page": page_counts_dict,
            }

            total_duration = time.time() - start_time
            print(f"[TIMELOG] Total function execution took: {total_duration:.2f} seconds")
            return final_result

        # Fallback for unexpected format
        total_duration = time.time() - start_time
        print(f"[TIMELOG] Total function execution took (failed): {total_duration:.2f} seconds")
        return {"raw_output": response,
                "error": f"Unexpected output format from API. Expected a 6-item list, got {len(response)}."}

    except Exception as e:
        # Re-raise exceptions as ValueError for API failure
        total_duration = time.time() - start_time
        print(f"[TIMELOG] Total function execution took (exception): {total_duration:.2f} seconds")
        raise ValueError(f"API call failed during prediction phase: {e}")

    finally:
        # 5. Clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                print(f"[WARN] Failed to delete temporary file {tmp_path}: {e}")


from gradio_client import Client


def get_grading_report(kb_text, question_text, answer_text):
    try:
        # Initialize client
        client = Client("heerjtdev/answer_validator")

        # USE THE API NAME FOUND: "/evaluate"
        result = client.predict(
            kb=kb_text,
            question=question_text,
            answer=answer_text,
            api_name="/evaluate"  # <--- Changed from /predict to /evaluate
        )

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}


from gradio_client import Client, handle_file
import os


def grade_student_answer(question: str, student_answer: str, context_text: str = None, file_path: str = None,
                         max_marks: int = 5):
    """
    Connects to the AI Grader Hugging Face Space to index content and grade an answer.

    Args:
        question (str): The question to ask.
        student_answer (str): The student's response.
        context_text (str, optional): Raw text content for the knowledge base.
        file_path (str, optional): Path to a local .pdf or .txt file.
        max_marks (int): Maximum score for the question.

    Returns:
        dict: A dictionary containing 'status', 'evidence', and 'feedback'.
    """

    # 1. Validation: Ensure we have exactly one source (Text OR File)
    if context_text and file_path:
        return {"error": "Please provide EITHER context_text OR file_path, not both."}
    if not context_text and not file_path:
        return {"error": "No content provided. Please provide context_text or file_path."}

    client_url = "https://heerjtdev-try-answer.hf.space"

    try:
        print(f"🔌 Connecting to {client_url}...")
        client = Client(client_url)

        # 2. Step 1: Index Content
        # We prepare the arguments based on what was provided
        if file_path:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            # Use handle_file for file uploads in Gradio Client
            idx_input_file = handle_file(file_path)
            idx_input_text = ""  # Empty string for text arg
            print(f"📤 Uploading file: {file_path}")
        else:
            idx_input_file = None
            idx_input_text = context_text
            print("📝 Indexing raw text...")

        # Call /process_content
        index_status = client.predict(
            file_obj=idx_input_file,
            raw_text=idx_input_text,
            api_name="/process_content"
        )

        # Check if indexing returned an error message string
        if "Error" in index_status or "content empty" in index_status.lower():
            return {"error": f"Indexing Failed: {index_status}"}

        # 3. Step 2: Retrieve & Grade
        print("🧠 Grading answer...")
        result = client.predict(
            question=question,
            student_answer=student_answer,
            max_marks=max_marks,
            api_name="/process_query"
        )

        evidence, feedback = result

        return {
            "success": True,
            "indexing_status": index_status,
            "evidence_used": evidence,
            "grading_feedback": feedback
        }

    except Exception as e:
        return {"success": False, "error": str(e)}











def extract_text_from_image(image_path):
    """
    Sends an image to the 'iammraat/laststraw' Space and returns the extracted text.

    Args:
        image_path (str): The local file path to the image.

    Returns:
        str: The extracted text, or None if an error occurs.

    """

    ocr_client = Client("iammraat/laststraw")

    try:
        # The API returns a tuple: (result_image_path, result_text)
        result = ocr_client.predict(
            image=handle_file(image_path),
            api_name="/run_pipeline"
        )

        # We only want the text (the second item in the tuple)
        extracted_text = result[1]

        return extracted_text

    except Exception as e:
        print(f"OCR Error: {e}")
        return None


# ==========================================
# Example Usage (Simulating your future API)
# ==========================================
if __name__ == "__main__":
    # CASE A: Using Text
    print("\n--- TEST CASE A: TEXT INPUT ---")
    response_text = grade_student_answer(
        question="Which Greek epic is Ulysses based on?",
        student_answer="It is based on Homer's Odyssey.",
        context_text="James Joyce's novel Ulysses is a parallel to Homer's Odyssey.",
        max_marks=5
    )
    print(response_text)


