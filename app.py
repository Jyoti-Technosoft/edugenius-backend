import uuid
from collections import Counter
import pickle
from typing import Dict, Any, Tuple, List
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from qdrant_client import QdrantClient, models
import os
import json
from datetime import datetime
import random
# from gradio_api import call_layoutlm_api
from gradio_api import call_yolo_api,latex_model, call_feeedback_api, get_grading_report, grade_student_answer, extract_text_from_image


















ADMIN_USER_ID = "vabtoa3ri7e9juu3cg33vzmw9cs2"


def is_admin(user_id):
    return str(user_id).strip().lower() == ADMIN_USER_ID.lower()


"""
===========================================================


MODEL OPTIONS


===========================================================
"""
app = Flask(__name__)

CORS(app)
# CORS(
#     app,
#     resources={r"/*": {
#         "origins": "https://edugenius-n679.onrender.com"
#     }},
#     supports_credentials=True,
#     allow_headers=["Content-Type", "Authorization"],
#     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
# )

# Simplified, more robust configuration
# CORS(app,
#      origins=["https://edugenius-n679.onrender.com"],
#      supports_credentials=True,
#      allow_headers=["Content-Type", "Authorization"],
#      methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

from collections import OrderedDict

"""
====================================================================

Helper Functions

====================================================================
"""

from vector_db import store_mcqs, fetch_mcqs, fetch_random_mcqs, store_test_session, fetch_test_by_testId, \
    test_sessions_by_userId, store_submitted_test, submitted_tests_by_userId, add_single_question, \
    update_single_question, delete_single_question, store_mcqs_for_manual_creation, delete_mcq_bank, \
    delete_submitted_test_by_id, delete_test_session_by_id, update_test_session, update_question_bank_metadata, \
    fetch_submitted_test_by_testId, delete_submitted_test_attempt, update_answer_flag_in_qdrant, normalize_answer,fetch_question_banks_metadata, fetch_question_context, client, COLLECTION_SUBMITTED, embed, _extract_payload, add_subscription_record, fetch_subscribed_questions, toggle_bank_public_status, fetch_public_marketplace, update_user_metadata_in_qdrant, fetch_community_marketplace, initialize_bank_record, fetch_user_flashcards, store_source_material, delete_source_material, fetch_user_sources, fetch_full_source_text


from werkzeug.utils import secure_filename


def format_mcq(mcq):
    return {
        "question": mcq.get("question") or mcq.get("ques") or mcq.get("q"),
        "noise": mcq.get("noise"),
        "image": mcq.get("image") or mcq.get("img"),
        "options": mcq.get("options") or mcq.get("opts"),
        "answer": mcq.get("answer") or mcq.get("ans") or mcq.get("correct")
    }
# ===================================================
# uncomment the text below to use gemini pipeline instead of the pre-trained model
# ===================================================

class Vocab:
    """Vocabulary class for serialization and lookup."""

    def __init__(self, min_freq=1, unk_token="<UNK>", pad_token="<PAD>"):
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.freq = Counter()
        self.itos = []
        self.stoi = {}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        """Allows lookup using word_vocab[token]. Returns UNK index if token is not found."""
        # Returns the index of the token, or the index of <UNK> if not found.
        return self.stoi.get(token, self.stoi[self.unk_token])

    # Methods for pickle serialization
    def __getstate__(self):
        return {
            'min_freq': self.min_freq,
            'unk_token': self.unk_token,
            'pad_token': self.pad_token,
            'itos': self.itos,
            'stoi': self.stoi,
        }

    def __setstate__(self, state):
        self.min_freq = state['min_freq']
        self.unk_token = state['unk_token']
        self.pad_token = state['pad_token']
        self.itos = state['itos']
        self.stoi = state['stoi']
        self.freq = Counter()


def load_vocabs(path: str) -> Tuple[Vocab, Vocab]:
    """Loads word and character vocabularies from a pickle file."""
    try:
        with open(path, "rb") as f:
            word_vocab, char_vocab = pickle.load(f)

        if len(word_vocab) <= 2 or len(char_vocab) <= 2:
            raise IndexError("Vocabulary file loaded but sizes are suspiciously small.")

        return word_vocab, char_vocab
    except FileNotFoundError:
        raise FileNotFoundError(f"Vocab file not found at {path}. Please check the path.")
    except Exception as e:
        raise RuntimeError(f"Error loading vocabs from {path}: {e}")


UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==============================
#
# API
#
# ===============================

# @app.route("/question-banks/upload", methods=["POST"])
# def upload_pdf():
#     print(f"\n[START] /create_question_bank request received")
#
#     # 1. Validate inputs
#     user_id = request.form.get("userId")
#     user_name = request.form.get("userName", "User")
#     title = request.form.get("title")
#     description = request.form.get("description")
#     pdf_file = request.files.get("pdf")
#
#     print(f"[INFO] Received form-data: userId={user_id}, title={title}, description={description}")
#     if not pdf_file:
#         return jsonify({"error": "PDF file not provided"}), 400
#
#     if not all([user_id, title, description]):
#         return jsonify({"error": "userId, title, description are required"}), 400
#
#     # 2. Keep PDF in memory (no Drive)
#     print("[STEP] Reading PDF into memory...")
#     pdf_bytes = pdf_file.read()
#     pdf_name = secure_filename(pdf_file.filename)
#
#     # 3. Directly call model
#     print("[STEP] Calling LayoutLM model directly (no Drive)...")
#     # final_data = call_layoutlm_api(pdf_bytes, pdf_name)
#     try:
#
#         final_data = latex_model(pdf_bytes, pdf_name)
#     except Exception as e:
#         print("[ERROR] latex_model failed → switching to YOLO model")
#         print("Reason:", e)
#         final_data = call_yolo_api(pdf_bytes, pdf_name)
#
#     # 4. Add index to MCQs
#     indexed_mcqs = [
#         {
#             **mcq,
#             "documentIndex": i,
#             "questionId": str(uuid.uuid4())  # ✅ assign unique ID
#         }
#         for i, mcq in enumerate(final_data)
#     ]
#
#     # 5. Store in vector DB
#     print("[STEP] Storing Question Bank in vector database...")
#     createdAtTimestamp = datetime.now().isoformat()
#     stored_id, all_have_answers = store_mcqs(
#         user_id,user_name, title, description, indexed_mcqs, pdf_name, createdAtTimestamp
#     )
#     print(f"[SUCCESS] Stored with generatedQAId={stored_id}")
#
#     print("[END] Request complete\n")
#     return Response(
#         json.dumps({
#             "generatedQAId": stored_id,
#             "userId": user_id,
#             "userName": user_name,
#             "fileName": pdf_name,
#             "createdAt": createdAtTimestamp,
#             "answerFound": all_have_answers
#         }, ensure_ascii=False),
#         mimetype="application/json"
#     )


import threading
import uuid
import json
from flask import Flask, request, jsonify, Response
from datetime import datetime
from werkzeug.utils import secure_filename



# This acts as a temporary status tracker
# In a real production app, use a database or Redis
processing_tasks = {}


def background_task(job_id, user_id, user_name, title, description, pdf_bytes, pdf_name):
    """The heavy lifting happens here in a separate thread."""
    try:
        print(f"[THREAD START] Processing job: {job_id}")

        # 3. Directly call model
        print("[STEP] Calling LayoutLM model directly...")
        try:
            final_data = latex_model(pdf_bytes, pdf_name)
        except Exception as e:
            print("[ERROR] latex_model failed → switching to YOLO model")
            print("Reason:", e)
            final_data = call_yolo_api(pdf_bytes, pdf_name)

        # 4. Add index to MCQs
        indexed_mcqs = [
            {
                **mcq,
                "documentIndex": i,
                "questionId": str(uuid.uuid4())
            }
            for i, mcq in enumerate(final_data)
        ]

        # 5. Store in vector DB
        print("[STEP] Storing Question Bank in vector database...")
        createdAtTimestamp = datetime.now().isoformat()

        # This is where your 2-minute delay happens
        stored_id, all_have_answers = store_mcqs(
            user_id, user_name, title, description, indexed_mcqs, pdf_name, createdAtTimestamp
        )

        # SAVE THE RESULT so the status endpoint can find it
        processing_tasks[job_id] = {
            "status": "completed",
            "generatedQAId": stored_id,
            "userId": user_id,
            "userName": user_name,
            "fileName": pdf_name,
            "createdAt": createdAtTimestamp,
            "answerFound": all_have_answers
        }
        print(f"[THREAD SUCCESS] Job {job_id} stored with id={stored_id}")

    except Exception as e:
        print(f"[THREAD ERROR] Job {job_id} failed: {str(e)}")
        processing_tasks[job_id] = {"status": "failed", "error": str(e)}


@app.route("/question-banks/upload", methods=["POST"])
def upload_pdf():
    print(f"\n[START] /question-banks/upload request received")

    # 1. Validate inputs
    user_id = request.form.get("userId")
    user_name = request.form.get("userName", "User")
    title = request.form.get("title")
    description = request.form.get("description")
    pdf_file = request.files.get("pdf")

    if not pdf_file:
        return jsonify({"error": "PDF file not provided"}), 400

    if not all([user_id, title, description]):
        return jsonify({"error": "userId, title, description are required"}), 400

    # 2. Prepare data for the thread
    pdf_bytes = pdf_file.read()
    pdf_name = secure_filename(pdf_file.filename)
    job_id = str(uuid.uuid4())  # Unique ID for Flutter to track

    # Initialize the status
    processing_tasks[job_id] = {"status": "processing"}

    # 🚀 START THREAD
    # We pass all the data the thread needs to finish the job
    thread = threading.Thread(
        target=background_task,
        args=(job_id, user_id, user_name, title, description, pdf_bytes, pdf_name)
    )
    thread.start()

    # RETURN IMMEDIATELY
    # Flutter gets this in 100ms and knows the work has started.
    print(f"[INFO] Thread started. Returning jobId: {job_id}")
    return jsonify({
        "status": "processing",
        "jobId": job_id,
        "message": "File received and processing started."
    }), 202


@app.route("/question-banks/status/<job_id>", methods=["GET"])
def get_status(job_id):
    """Flutter will call this every 5-10 seconds to check if it's done."""
    task = processing_tasks.get(job_id)

    if not task:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(task)



@app.route("/document-analysis", methods=["POST"])
def analyze_pdf():
    print(f"\n[START] /document-analysis request received")

    # 1. Validate inputs
    pdf_file = request.files.get("pdf")
    if not pdf_file:
        return jsonify({"error": "PDF file not provided"}), 400

    # 2. Keep PDF in memory
    print("[STEP] Reading PDF into memory...")
    pdf_bytes = pdf_file.read()
    pdf_name = secure_filename(pdf_file.filename)

    # 3. Call Feedback API for analysis
    print("[STEP] Calling feedback model for document analysis...")

    # Initialize variables, including the new dictionary
    total_pages = 0
    total_equations = 0
    equation_counts_per_page = {}  # <-- NEW: Initialize the variable

    try:
        # Call the function with in-memory bytes and filename
        feedback_data: Dict[str, Any] = call_feeedback_api(pdf_bytes, pdf_name)

        # Safely extract the returned values
        total_pages = feedback_data.get('Total Pages in PDF', 0)
        total_equations = feedback_data.get('Total Equations Detected', 0)

        # <-- CRITICAL FIX: Extract the new per-page data
        equation_counts_per_page = feedback_data.get('Equation Counts Per Page', {})

        print(f"[INFO] Analysis results: Pages={total_pages}, Equations={total_equations}")

        # 4. Return the analysis data immediately
        print("[END] Analysis request complete\n")
        return jsonify({
            "status": "success",
            "totalPages": total_pages,
            "totalEquations": total_equations,
            "fileName": pdf_name,  # Include filename for continuity
            # <-- CRITICAL FIX: Include the new data in the final JSON response
            "equationCountsPerPage": equation_counts_per_page
        })

    except Exception as e:
        print(f"[ERROR] call_feeedback_api failed: {e}")
        return jsonify({"error": f"Document analysis failed: {e}"}), 500


#
# @app.route("/question-banks/upload/images", methods=["POST"])
# def upload_image():
#     print("\n[START] /create_question_bank request received")
#
#     # 1. Validate inputs
#     user_id = request.form.get("userId")
#     user_name = request.form.get("userName", "User")
#     title = request.form.get("title")
#     description = request.form.get("description")
#     image_files = request.files.getlist("image")  # ✅ multiple images
#
#     print(f"[INFO] Received form-data: userId={user_id}, title={title}, description={description}")
#     if not image_files or len(image_files) == 0:
#         return jsonify({"error": "No image file(s) provided"}), 400
#
#     if not all([user_id, title, description]):
#         return jsonify({"error": "userId, title, description are required"}), 400
#
#     all_results = []
#
#     # 2. Loop through each image
#     for idx, img_file in enumerate(image_files, start=1):
#         print(f"[STEP] Reading image {idx}/{len(image_files)} into memory...")
#         file_bytes = img_file.read()
#         filename = secure_filename(img_file.filename)
#
#         # 3. Directly call model for each image
#         print(f"[STEP] Calling LayoutLM model for {filename} ...")
#         try:
#             result = latex_model(file_bytes, filename)
#             print(f"[SUCCESS] Model returned result for {filename}")
#             if isinstance(result, list):
#                 all_results.extend(result)
#             else:
#                 all_results.append(result)
#         except Exception as e:
#             print(f"[ERROR] Failed on {filename}: {e}")
#
#     # 4. Add index to MCQs
#     indexed_mcqs = [
#         {**mcq, "documentIndex": i}
#         for i, mcq in enumerate(all_results)
#     ]
#
#     # 5. Store in vector DB
#     print("[STEP] Storing Question Bank in vector database...")
#     createdAtTimestamp = datetime.now().isoformat()
#     stored_id = store_mcqs(
#         user_id, user_name, title, description, indexed_mcqs, "multiple_images.zip", createdAtTimestamp
#     )
#     print(f"[SUCCESS] Stored with generatedQAId={stored_id}")
#
#     print("[END] Request complete\n")
#     return Response(
#         json.dumps({
#             "generatedQAId": stored_id,
#             "userId": user_id,
#             "fileCount": len(image_files),
#             "createdAt": createdAtTimestamp,
#         }, ensure_ascii=False),
#         mimetype="application/json"
#     )


import threading
import uuid
import json
from flask import Flask, request, jsonify, Response
from datetime import datetime
from werkzeug.utils import secure_filename


# Reuse the same status dictionary we set up for the PDF logic
# processing_tasks = {}

def background_image_task(job_id, user_id, user_name, title, description, image_data_list):
    """Heavy lifting for multiple images happens here."""
    try:
        print(f"[THREAD START] Processing Image Job: {job_id}")
        all_results = []

        # 2. Loop through each image (data already in memory as bytes)
        for idx, (file_bytes, filename) in enumerate(image_data_list, start=1):
            print(f"[STEP] Processing image {idx}/{len(image_data_list)}: {filename}")

            try:
                # Call your AI model
                result = latex_model(file_bytes, filename)
                print(f"[SUCCESS] Model returned result for {filename}")

                if isinstance(result, list):
                    all_results.extend(result)
                else:
                    all_results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed on {filename}: {e}")

        # 4. Add index to MCQs
        indexed_mcqs = [
            {**mcq, "documentIndex": i, "questionId": str(uuid.uuid4())}
            for i, mcq in enumerate(all_results)
        ]

        # 5. Store in vector DB
        print("[STEP] Storing Question Bank in vector database...")
        createdAtTimestamp = datetime.now().isoformat()

        # Note: Using your existing store_mcqs function
        stored_id, all_have_answers = store_mcqs(
            user_id, user_name, title, description, indexed_mcqs, "multiple_images.zip", createdAtTimestamp
        )

        # Update the global task status so Flutter polling finds it
        processing_tasks[job_id] = {
            "status": "completed",
            "generatedQAId": stored_id,
            "userId": user_id,
            "userName": user_name,
            "fileCount": len(image_data_list),
            "createdAt": createdAtTimestamp,
            "answerFound": all_have_answers
        }
        print(f"[THREAD SUCCESS] Image Job {job_id} stored with id={stored_id}")

    except Exception as e:
        print(f"[THREAD ERROR] Image Job {job_id} failed: {str(e)}")
        processing_tasks[job_id] = {"status": "failed", "error": str(e)}


@app.route("/question-banks/upload/images", methods=["POST"])
def upload_image():
    print("\n[START] /question-banks/upload/images request received")

    # 1. Validate inputs
    user_id = request.form.get("userId")
    user_name = request.form.get("userName", "User")
    title = request.form.get("title")
    description = request.form.get("description")
    image_files = request.files.getlist("image")

    if not image_files or len(image_files) == 0:
        return jsonify({"error": "No image file(s) provided"}), 400

    if not all([user_id, title, description]):
        return jsonify({"error": "userId, title, description are required"}), 400

    # Pre-process files into memory before starting thread
    # We must do this because request.files is not thread-safe in Flask
    image_data_list = []
    for img_file in image_files:
        content = img_file.read()
        fname = secure_filename(img_file.filename)
        image_data_list.append((content, fname))

    job_id = str(uuid.uuid4())
    processing_tasks[job_id] = {"status": "processing"}

    # START THREAD
    thread = threading.Thread(
        target=background_image_task,
        args=(job_id, user_id, user_name, title, description, image_data_list)
    )
    thread.start()

    print(f"[INFO] Thread started for {len(image_data_list)} images. jobId: {job_id}")

    # Return 202 immediately so Flutter starts polling
    return jsonify({
        "status": "processing",
        "jobId": job_id,
        "message": f"Processing {len(image_data_list)} images in background."
    }), 202


# @app.route("/question-banks", methods=["GET"])
# def get_question_banks_by_user():
#     userId = request.args.get("userId")  # ✅ GET query params
#
#     if not userId:
#         return jsonify({"error": "userId is required"}), 400
#
#     mcqs_data = fetch_mcqs(userId=userId)
#     if not mcqs_data:
#         return jsonify({"message": "No Paper Sets found"})
#
#     # FIX: Iterate through each paper set and sort its MCQs list
#     for paper_set in mcqs_data:
#         # Check if the 'mcqs' list exists and is iterable
#         if paper_set.get('metadata', {}).get('mcqs'):
#             mcqs_list = paper_set['metadata']['mcqs']
#
#             # This handles older data that might have missing or None 'documentIndex' values.
#             paper_set['metadata']['mcqs'] = sorted(
#                 mcqs_list,
#                 key=lambda x: int(x['documentIndex'])
#                 if x.get('documentIndex') is not None else float('inf')
#             )
#             # ===============================================
#
#     return Response(
#         json.dumps(mcqs_data, ensure_ascii=False, indent=4),
#         mimetype="application/json"
#     )
#





@app.route("/question-banks", methods=["GET"])
def get_question_banks_by_user():
    userId = request.args.get("userId")
    if not userId:
        return jsonify({"error": "userId is required"}), 400

    # Normalize case like in store_mcqs
    userId = userId.strip().lower()

    banks = fetch_question_banks_metadata(userId)
    if not banks:
        return jsonify({"message": "No Question Banks found"})

    return jsonify(banks)



@app.route("/question-banks/<generatedQAId>", methods=["GET"])
def get_question_bank_by_id(generatedQAId):
    page = request.args.get('page', default=1, type=int)
    limit = request.args.get('limit', default=10000, type=int)

    # Change 'page_size' to 'limit' here to match your function definition
    result = fetch_mcqs(generatedQAId=generatedQAId, page=page, limit=limit)

    if not result:
        return jsonify({"error": "No MCQs found"}), 404

    return jsonify(result)


@app.route("/tests", methods=["POST"])
def generate_test():
    """
    API to fetch MCQs by generated-qa-Id and marks (limit),
    and also to create a new test entry.
    """
    data = request.get_json(silent=True) or request.form

    generatedQAId = data.get("generatedQAId")
    marks = data.get("marks")
    userId = data.get("userId")
    testTitle = data.get("testTitle")
    totalTime = data.get("totalTime")

    if not generatedQAId:
        return jsonify({"error": "generatedQAId is required"}), 400
    # ... (other validation checks)

    try:
        marks = int(marks)
    except ValueError:
        return jsonify({"error": "marks must be an integer"}), 400

    testId = str(uuid.uuid4())
    createdAt = datetime.now().isoformat()

    # 1. Fetch random sample
    test_data_results = fetch_random_mcqs(generatedQAId, num_questions=marks)

    if not test_data_results:
        return jsonify({"message": "No MCQs found"}), 200

    mcqs_data = test_data_results[0].get("metadata", {}).get("mcqs", [])

    # The list mcqs_data is now in the final, random order for the test.

    # 2. ASSIGN NEW SEQUENTIAL INDEX (testIndex)
    final_mcqs_for_storage = []
    for i, mcq in enumerate(mcqs_data):
        # Assign a sequential index starting from 1 for the client/storage
        mcq['testIndex'] = i + 1
        final_mcqs_for_storage.append(mcq)

    # 3. Store the session using the indexed list
    if userId:
        is_stored = store_test_session(userId, testId, testTitle, totalTime, createdAt, final_mcqs_for_storage)
        if not is_stored:
            return jsonify({"error": "Failed to store test session"}), 500

    # 4. Return the result
    return jsonify({
        "message": "Test created and stored successfully",
        "userId": userId,
        "testId": testId,
        "totalTime": totalTime,
        "createdAt": createdAt,
        "questions": final_mcqs_for_storage  # Return the indexed list
    }), 200


@app.route("/tests/combined", methods=["POST"])
def combined_test():
    data = request.get_json(silent=True) or request.form

    userId = data.get("userId")
    testTitle = data.get("testTitle")
    totalTime = data.get("totalTime")
    total_questions = data.get("total_questions")
    sources = data.get("sources")

    # Validate required inputs
    if not all([userId, testTitle, totalTime, total_questions, sources]) or not isinstance(sources, list):
        return jsonify(
            {"error": "userId, testTitle, total_questions, totalTime, and a list of sources are required"}), 400

    try:
        total_questions = int(total_questions)
        if sum(s.get("percentage", 0) for s in sources) != 100:
            return jsonify({"error": "Percentages must sum to 100"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "total_questions must be an integer and percentages must be numbers"}), 400

    all_mcqs = []

    for source in sources:
        qa_id = source.get("generatedQAId")
        percentage = source.get("percentage")

        if not qa_id or not percentage:
            return jsonify({"error": "Each source must have 'generatedQAId' and 'percentage'"}), 400

        # Calculate the number of questions for this source
        num_questions = round(total_questions * (percentage / 100))

        # Fetch a random sample from this source
        # Note: fetch_random_mcqs returns a list containing a dict with metadata/mcqs
        mcqs_record = fetch_random_mcqs(generatedQAId=qa_id, num_questions=num_questions)

        if mcqs_record:
            # Extract the list of questions and combine them
            all_mcqs.extend(mcqs_record[0].get("metadata", {}).get("mcqs", []))

    # Shuffle the combined list of all MCQs to finalize the test order
    random.shuffle(all_mcqs)

    if not all_mcqs:
        return jsonify({"message": "No MCQs found for the provided IDs"}), 200

    # Assign a new, sequential index (testIndex) to each question
    final_mcqs_for_storage = []
    for i, mcq in enumerate(all_mcqs):
        # Assign a sequential index starting from 1
        mcq['testIndex'] = i + 1
        final_mcqs_for_storage.append(mcq)

    # Generate test metadata
    testId = str(uuid.uuid4())
    createdAt = datetime.now().isoformat()

    # Store the test session with the indexed list
    store_test_session(userId, testId, testTitle, totalTime, createdAt, final_mcqs_for_storage)

    return jsonify({
        "userId": userId,
        "testId": testId,
        "testTitle": testTitle,
        "totalTime": totalTime,
        "createdAt": createdAt,
        "questions": final_mcqs_for_storage  # Return the correctly indexed list
    }), 200


@app.route("/tests/<testId>", methods=["GET"])
def testId(testId):
    """

    API to fetch a specific test session by its ID.
    """
    test_data = fetch_test_by_testId(testId)
    if not test_data:
        return jsonify({"error": "Test  not found"}), 200
    for q in test_data.get("questions", []):
        q.pop("answer", None)

    return jsonify(test_data), 200


@app.route("/paper-sets/user/<userId>", methods=["GET"])
def test_history_by_userId(userId):
    test_history = test_sessions_by_userId(userId)
    if not test_history:
        return jsonify({"message": "No test sessions found"}), 200

    # remove answers before sending to frontend
    for test in test_history:
        for q in test.get("questions", []):
            q.pop("answer", None)  # removes if present

    return jsonify(test_history), 200





#
# @app.route("/tests/submit", methods=["POST"])
# def submit_test():
#     """
#     Enhanced API to submit student answers.
#     Calculates overall score and breaks down performance by Subject and Concept.
#     """
#     data = request.get_json(silent=True) or {}
#     userId = data.get("userId")
#     testId = data.get("testId")
#     testTitle = data.get("testTitle")
#     timeSpent = data.get("timeSpent")
#     totalTime = data.get("totalTime")
#     answers = data.get("answers")
#
#     if not all([userId, testId, answers]):
#         return jsonify({"error": "Missing required fields: userId, testId, answers"}), 400
#     if not isinstance(answers, list):
#         return jsonify({"error": "Answers must be a list"}), 400
#
#     submittedAt = datetime.now().isoformat()
#
#     # 🧠 Fetch original test data to verify correct answers and metadata
#     test_data = fetch_test_by_testId(testId)
#     if not test_data:
#         return jsonify({"error": "Test not found"}), 404
#
#     questions = test_data.get("questions", [])
#     if isinstance(questions, str):
#         try:
#             questions = json.loads(questions)
#         except Exception:
#             questions = []
#
#     # Build lookup for questions by ID
#     question_map = {q.get("questionId"): q for q in questions if q.get("questionId")}
#
#     totalQuestions = len(questions)
#     total_correct = 0
#     total_mcq = 0
#     total_descriptive = 0
#     mcq_correct = 0
#     results = []
#     descriptive_question_ids = []
#
#     # 📊 Performance Analysis Tracking
#     subject_analysis = {}  # { "Physics": {"total": 0, "correct": 0} }
#     concept_analysis = {}  # { "Equilibrium": {"total": 0, "correct": 0} }
#
#     # ✅ Process each submitted answer
#     for ans in answers:
#         qid = ans.get("questionId")
#         qtext = ans.get("question")
#         user_ans = ans.get("your_answer")
#
#         # Find the question details in the master map
#         question_details = question_map.get(qid)
#
#         # Fallback to text matching if ID is missing
#         if not question_details and qtext:
#             for q in questions:
#                 if qtext.strip().lower() == q.get("question", "").strip().lower():
#                     question_details = q
#                     qid = q.get("questionId")
#                     break
#
#         if not question_details:
#             results.append(OrderedDict([
#                 ("questionId", qid),
#                 ("question", qtext),
#                 ("status", "question_not_found")
#             ]))
#             continue
#
#         # Extract Subject and Concept Labels
#         subj_label = (question_details.get("predicted_subject") or {}).get("label", "General")
#         conc_label = (question_details.get("predicted_concept") or {}).get("label", "General")
#
#         # Initialize analysis containers
#         if subj_label not in subject_analysis:
#             subject_analysis[subj_label] = {"total": 0, "correct": 0}
#         if conc_label not in concept_analysis:
#             concept_analysis[conc_label] = {"total": 0, "correct": 0}
#
#         question_type = (question_details.get("question_type") or "MCQ").upper()
#
#         if question_type == "MCQ":
#             total_mcq += 1
#             subject_analysis[subj_label]["total"] += 1
#             concept_analysis[conc_label]["total"] += 1
#
#             correct_ans = question_details.get("answer")
#             is_correct = (normalize_answer(user_ans) == normalize_answer(correct_ans))
#
#             if is_correct:
#                 total_correct += 1
#                 mcq_correct += 1
#                 subject_analysis[subj_label]["correct"] += 1
#                 concept_analysis[conc_label]["correct"] += 1
#
#             results.append(OrderedDict([
#                 ("questionId", qid),
#                 ("question", question_details.get("question", "")),
#                 ("subject", subj_label),
#                 ("concept", conc_label),
#                 ("question_type", "MCQ"),
#                 ("your_answer", user_ans),
#                 ("correct_answer", correct_ans),
#                 ("is_correct", is_correct),
#                 ("status", "graded")
#             ]))
#
#         else:  # DESCRIPTIVE
#             total_descriptive += 1
#             subject_analysis[subj_label]["total"] += 1
#             concept_analysis[conc_label]["total"] += 1
#             descriptive_question_ids.append(qid)
#
#             results.append(OrderedDict([
#                 ("questionId", qid),
#                 ("question", question_details.get("question", "")),
#                 ("subject", subj_label),
#                 ("concept", conc_label),
#                 ("question_type", "DESCRIPTIVE"),
#                 ("your_answer", user_ans),
#                 ("correct_answer", question_details.get("knowledge_base", "")),
#                 ("is_correct", None),  # Pending AI grading
#                 ("status", "pending_grading")
#             ]))
#
#     # 🧮 Calculate preliminary score (from MCQs)
#     preliminary_score = round((total_correct / totalQuestions * 100), 2) if totalQuestions > 0 else 0.0
#
#     # 💾 Store submission attempt in DB
#     is_stored, attemptId = store_submitted_test(
#         userId=userId,
#         testId=testId,
#         testTitle=testTitle,
#         timeSpent=timeSpent,
#         totalTime=totalTime,
#         submittedAt=submittedAt,
#         detailed_results=results,
#         score=preliminary_score,
#         total_questions=totalQuestions,
#         total_correct=total_correct,
#         total_mcq=total_mcq,
#         total_descriptive=total_descriptive,
#         mcq_correct=mcq_correct,
#         subject_analysis=subject_analysis,  # NEW
#         concept_analysis=concept_analysis,  # NEW
#         grading_status="partial" if total_descriptive > 0 else "complete"
#     )
#
#     if not is_stored:
#         return jsonify({"error": "Failed to store submission"}), 500
#
#     # 📦 Final response with full analysis
#     response = OrderedDict([
#         ("attemptId", attemptId),
#         ("userId", userId),
#         ("testId", testId),
#         ("testTitle", testTitle),
#         ("submittedAt", submittedAt),
#         ("timeSpent", timeSpent),
#         ("score", preliminary_score),
#         ("grading_status", "partial" if total_descriptive > 0 else "complete"),
#         ("performance_report", {
#             "by_subject": subject_analysis,
#             "by_concept": concept_analysis
#         }),
#         ("stats", {
#             "total_questions": totalQuestions,
#             "total_mcq": total_mcq,
#             "total_descriptive": total_descriptive,
#             "mcq_correct": mcq_correct
#         }),
#         ("descriptive_pending", descriptive_question_ids),
#         ("detailed_results", results)
#     ])
#
#     return jsonify(response)

@app.route("/tests/submit", methods=["POST"])
def submit_test():
    """
    Final Submission Endpoint.
    - Matches User's JSON structure.
    - Uses 'linkedSourceId' to find context.
    - Grades MCQs numerically.
    - Descriptive/HTR: Stores AI textual feedback without affecting score.
    """
    try:
        # 1. Parse JSON Data
        data = request.get_json(silent=True) or {}

        userId = data.get("userId")
        testId = data.get("testId")
        testTitle = data.get("testTitle")
        timeSpent = data.get("timeSpent")
        totalTime = data.get("totalTime")
        answers = data.get("answers", [])

        if not all([userId, testId]):
            return jsonify({"error": "Missing required fields: userId, testId"}), 400

        submittedAt = datetime.now().isoformat()

        # 2. Fetch Test Metadata (The JSON you provided above comes from here)
        test_data = fetch_test_by_testId(testId)
        if not test_data:
            return jsonify({"error": "Test not found"}), 404

        questions = test_data.get("questions", [])
        if isinstance(questions, str):
            questions = json.loads(questions)

        # Map for O(1) lookup
        question_map = {str(q.get("questionId")): q for q in questions if q.get("questionId")}

        # 3. Grading & Analysis Containers
        totalQuestions = len(questions)
        total_correct = 0
        total_mcq = 0
        total_descriptive = 0
        mcq_correct = 0

        results = []
        ai_feedback_report = {}
        subject_analysis = {}
        concept_analysis = {}

        # 4. PROCESS ANSWERS
        for ans in answers:
            qid = str(ans.get("questionId"))
            user_ans = ans.get("your_answer", "")

            # Fetch Master Data from your JSON structure
            q_details = question_map.get(qid)
            if not q_details: continue

            # Extract Tags
            subj_label = (q_details.get("predicted_subject") or {}).get("label", "General")
            conc_label = (q_details.get("predicted_concept") or {}).get("label", "General")

            if subj_label not in subject_analysis: subject_analysis[subj_label] = {"total": 0, "correct": 0}
            if conc_label not in concept_analysis: concept_analysis[conc_label] = {"total": 0, "correct": 0}

            subject_analysis[subj_label]["total"] += 1
            concept_analysis[conc_label]["total"] += 1

            q_type = (q_details.get("question_type") or "MCQ").upper()

            # --- A. MCQ Logic ---
            if q_type == "MCQ":
                total_mcq += 1
                correct_ans = q_details.get("options", {}).get(
                    q_details.get("answer", ""))  # Handle option mapping if needed, or direct match
                # Fallback: if 'answer' is direct value (like '4')
                if not correct_ans:
                    correct_ans = q_details.get("answer")

                is_correct = (normalize_answer(str(user_ans)) == normalize_answer(str(correct_ans)))

                if is_correct:
                    total_correct += 1
                    mcq_correct += 1
                    subject_analysis[subj_label]["correct"] += 1
                    concept_analysis[conc_label]["correct"] += 1

                results.append(OrderedDict([
                    ("questionId", qid),
                    ("question", q_details.get("question", "")),
                    ("subject", subj_label),
                    ("concept", conc_label),
                    ("question_type", "MCQ"),
                    ("your_answer", user_ans),
                    ("correct_answer", correct_ans),
                    ("is_correct", is_correct),
                    ("status", "graded")
                ]))

            # --- B. Descriptive / HTR Logic ---
            else:
                total_descriptive += 1

                # ✅ CRITICAL FIX: Look for 'linkedSourceId' based on your JSON
                source_id = q_details.get("linkedSourceId") or q_details.get("sourceId")

                grading_status = "pending_grading"
                ai_feedback_text = "No feedback generated."

                # Check: Source ID + Answer exists?
                if source_id and user_ans and len(str(user_ans)) > 1:
                    print(f"[INFO] Fetching context for SourceID: {source_id}")
                    context_text = fetch_full_source_text(source_id)

                    if context_text:
                        formatted_context = f"STORY CONTEXT:\n{context_text}\n\nEND OF CONTEXT"
                        try:
                            # 1. CALL AI GRADER
                            ai_result = grade_student_answer(
                                question=q_details.get("question", ""),
                                student_answer=user_ans,
                                context_text=formatted_context,
                                max_marks=q_details.get("marks", 5)
                            )

                            # 2. STORE FEEDBACK
                            if ai_result.get("success"):
                                grading_status = "graded"
                                ai_feedback_text = ai_result.get("grading_feedback", "No feedback text.")
                                ai_feedback_report[qid] = ai_result
                            else:
                                print(f"[WARN] AI Grading failed for {qid}: {ai_result.get('error')}")

                        except Exception as e:
                            print(f"[ERROR] Auto-grading error QID {qid}: {e}")
                    else:
                        print(f"[WARN] No text found for SourceID: {source_id}")

                results.append(OrderedDict([
                    ("questionId", qid),
                    ("question", q_details.get("question", "")),
                    ("subject", subj_label),
                    ("concept", conc_label),
                    ("question_type", "DESCRIPTIVE"),
                    ("your_answer", user_ans),
                    ("correct_answer", "See AI Feedback"),
                    ("is_correct", None),
                    ("ai_score", 0),
                    ("ai_feedback", ai_feedback_text),
                    ("status", grading_status)
                ]))

        # 5. Final Calculations
        final_score = round((total_correct / totalQuestions * 100), 2) if totalQuestions > 0 else 0.0

        pending_count = sum(1 for r in results if r["status"] == "pending_grading")
        final_status = "partial" if pending_count > 0 else "complete"

        # 6. Store Submission
        is_stored, attemptId = store_submitted_test(
            userId=userId,
            testId=testId,
            testTitle=testTitle,
            timeSpent=timeSpent,
            totalTime=totalTime,
            submittedAt=submittedAt,
            detailed_results=results,
            score=final_score,
            total_questions=totalQuestions,
            total_correct=total_correct,
            total_mcq=total_mcq,
            total_descriptive=total_descriptive,
            mcq_correct=mcq_correct,
            subject_analysis=subject_analysis,
            concept_analysis=concept_analysis,
            grading_status=final_status,
            ai_feedback=ai_feedback_report
        )

        if not is_stored:
            return jsonify({"error": "Failed to store submission"}), 500

        return jsonify({
            "message": "Submission processed successfully",
            "attemptId": attemptId,
            "score": final_score,
            "grading_status": final_status,
            "results": results,
            "ai_feedback": ai_feedback_report
        }), 201

    except Exception as e:
        print(f"[ERROR] /tests/submit: {e}")
        return jsonify({"error": str(e)}), 500





@app.route("/tests/submitted/user/<userId>", methods=["GET"])
def submitted_tests_history(userId):
    """
    API to fetch a list of all submitted test sessions for a given user.
    """
    if not userId:
        return jsonify({"error": "userId is required"}), 400

    submitted_tests = submitted_tests_by_userId(userId)

    if submitted_tests is None:
        return jsonify({"error": "An error occurred while fetching submitted tests"}), 500

    if not submitted_tests:
        return jsonify({"message": "No submitted tests found for this user"}), 200

    return jsonify(submitted_tests), 200


@app.route("/tests/submitted/<testId>", methods=["GET"])
def get_single_submitted_test(testId):
    """
    Fetch details of one submitted test by testId.
    """
    if not testId:
        return jsonify({"error": "testId is required"}), 400

    result = fetch_submitted_test_by_testId(testId)

    if not result:
        return jsonify({"message": "No submitted test found"}), 404

    return jsonify(result), 200


@app.route("/question-banks/<generatedQAId>", methods=["PUT"])
def edit_question_bank(generatedQAId):
    """
    Unified API to perform add, edit, or delete operations on questions,
    and update the question bank's Title and Description.

    Accepts both:
    1. {
          "title": "English Test",
          "description": "Updated chapter 1 test",
          "edits": [ { "operation": "edit", "data": {...}} ]
       }
    2. [ { "operation": "edit", "data": {...}} ]  ← Legacy (frontend-only edits)
    """

    # Step 1: Parse request JSON
    payload = request.get_json(silent=True) or {}

    # Handle both dict and list payloads
    if isinstance(payload, list):
        edits = payload
        new_title = None
        new_description = None
        is_public = None
    else:
        edits = payload.get("edits")
        new_title = payload.get("title")
        new_description = payload.get("description")
        is_public = payload.get("public")

    metadata_update_status = {
        "title_updated": False,
        "description_updated": False,
        "success": True
    }

    # --- Step 2: Update Metadata (Title / Description) ---
    try:
        if new_title is not None or new_description is not None:
            metadata_update_status = update_question_bank_metadata(
                generatedQAId=generatedQAId,
                title=new_title,
                description=new_description,
                is_public=is_public

            )

            # Handle metadata update failure
            if not metadata_update_status.get("success", True):
                return jsonify({
                    "error": f"Failed to update metadata for Question Bank ID: {generatedQAId}"
                }), 500
    except Exception as e:
        print(f"[ERROR] Metadata update failed: {str(e)}")
        metadata_update_status["success"] = False

    # --- Step 3: Process Question-Level Edits ---
    if edits and isinstance(edits, list):
        for edit in edits:
            try:
                operation = edit.get("operation")
                data = edit.get("data")

                if not operation or not data:
                    continue

                if operation == "add":
                    add_single_question(generatedQAId, data)

                elif operation == "edit":
                    questionId = data.get("questionId")
                    if questionId:
                        update_single_question(questionId, data)

                elif operation == "delete":
                    questionId = data.get("questionId")
                    if questionId:
                        delete_single_question(questionId)

                else:
                    print(f"[WARN] Unknown operation '{operation}' ignored.")

            except Exception as e:
                print(f"[ERROR] Failed to process edit operation: {str(e)}")
                continue

    # --- Step 4: Fetch Updated Data for Response ---
    try:
        updated_data = fetch_mcqs(generatedQAId=generatedQAId)
    except Exception as e:
        print(f"[ERROR] Failed to fetch updated question bank: {str(e)}")
        updated_data = None

    if not updated_data:
        return jsonify({
            "error": "Update processed, but the question bank was not found.",
            "generatedQAId_used": generatedQAId
        }), 404

        # ✅ --- Step 5: Compute answerFound flag ---
    mcqs = updated_data[0]["metadata"].get("mcqs", [])
    all_have_answers = True
    for q in mcqs:
        ans = q.get("answer")
        if not (ans and str(ans).strip()):
            all_have_answers = False
            break

    # ✅ --- Step 6: Update Qdrant MCQ bank with answerFound flag ---
    update_answer_flag_in_qdrant(generatedQAId, all_have_answers)

    updated_questions_count = len(mcqs)

    # ✅ --- Step 7: Return Success Response ---
    return jsonify({
        "message": "Question bank updated successfully",
        "title_updated": metadata_update_status.get("title_updated", False),
        "description_updated": metadata_update_status.get("description_updated", False),
        "updated_questions_count": updated_questions_count,
        "answerFound": all_have_answers
    }), 200

#
# @app.route("/question-banks/manual", methods=["POST"])
# def create_manual_question_bank():
#     """
#     API to create a new question bank and populate it with a list of questions
#     in a single request for a smoother user experience.
#     """
#     data = request.get_json(silent=True) or request.form.to_dict()
#     user_id = data.get("userId")
#     title = data.get("title")
#     description = data.get("description")
#     raw_mcqs = data.get("questions", [])  # Expects a list of question objects
#
#     if not all([user_id, title, description]) or not isinstance(raw_mcqs, list):
#         return jsonify({"error": "userId, title, description, and a list of 'questions' are required"}), 400
#
#     if not raw_mcqs:
#         return jsonify({"error": "Question bank must contain at least one question."}), 400
#
#     indexed_mcqs = []
#
#     # 1. Format and Index MCQs (similar to your upload_pdf route logic)
#     for i, mcq in enumerate(raw_mcqs):
#         # Ensure options are properly formatted (if they come as a dict from the client)
#         if 'options' in mcq and isinstance(mcq['options'], dict):
#             # We need to ensure the options are stored as a JSON string
#             # as required by the ChromaDB metadata constraint (as discovered earlier).
#             mcq['options'] = json.dumps(mcq['options'])
#
#         # NOTE: If your database requires questionId/documentIndex, they must be set here.
#         # However, we will assume 'store_mcqs_for_manual_creation' handles questionId and documentIndex assignment.
#         mcq['documentIndex'] = i
#         mcq['questionId'] = str(uuid.uuid4())
#         indexed_mcqs.append(mcq)
#
#     # 2. Store Metadata and Questions (using a modified store function)
#     try:
#         # Create a function similar to store_mcqs but for manual data
#         generated_qa_id = store_mcqs_for_manual_creation(
#             user_id,
#             title,
#             description,
#             indexed_mcqs
#         )
#     except Exception as e:
#         print(f"Error storing manual question bank: {e}")
#         return jsonify({"error": "Failed to create and store question bank"}), 500
#
#     return jsonify({
#         "message": "Question bank created and populated successfully",
#         "generatedQAId": generated_qa_id,
#         "userId": user_id,
#         "title": title,
#         "questions_count": len(indexed_mcqs)
#     }), 201

#
# @app.route("/question-banks/manual", methods=["POST"])
# def create_manual_question_bank():
#     """
#     API to create a new question bank.
#     Updated to support 'linkedSourceId' for connecting questions to a specific PDF source.
#     """
#     data = request.get_json(silent=True) or request.form.to_dict()
#
#     # 1. Extract Fields
#     user_id = data.get("userId")
#     title = data.get("title")
#     description = data.get("description")
#     raw_mcqs = data.get("questions", [])
#
#     # --- NEW: Get the source ID if it exists ---
#     linked_source_id = data.get("linkedSourceId")
#     # -------------------------------------------
#
#     if not all([user_id, title, description]) or not isinstance(raw_mcqs, list):
#         return jsonify({"error": "userId, title, description, and a list of 'questions' are required"}), 400
#
#     if not raw_mcqs:
#         return jsonify({"error": "Question bank must contain at least one question."}), 400
#
#     indexed_mcqs = []
#
#     # 2. Format Questions
#     for i, mcq in enumerate(raw_mcqs):
#         if 'options' in mcq and isinstance(mcq['options'], dict):
#             mcq['options'] = json.dumps(mcq['options'])
#
#         mcq['documentIndex'] = i
#         mcq['questionId'] = str(uuid.uuid4())
#
#         # --- NEW: Tag individual questions with the source ID too (Optional but recommended) ---
#         if linked_source_id:
#             mcq['linkedSourceId'] = linked_source_id
#
#         indexed_mcqs.append(mcq)
#
#     # 3. Store Metadata and Questions
#     try:
#         # You need to update your store_mcqs_for_manual_creation function in vector_db.py
#         # to accept this new argument.
#         generated_qa_id = store_mcqs_for_manual_creation(
#             user_id,
#             title,
#             description,
#             indexed_mcqs,
#             linked_source_id=linked_source_id  # <--- PASS IT HERE
#         )
#     except Exception as e:
#         print(f"Error storing manual question bank: {e}")
#         return jsonify({"error": "Failed to create and store question bank"}), 500
#
#     return jsonify({
#         "message": "Question bank created successfully",
#         "generatedQAId": generated_qa_id,
#         "userId": user_id,
#         "title": title,
#         "linkedSourceId": linked_source_id,  # Return it so frontend confirms it's linked
#         "questions_count": len(indexed_mcqs)
#     }), 201


@app.route("/question-banks/manual", methods=["POST"])
def create_manual_question_bank():
    """
    API to create a new question bank manually.
    - Supports 'linkedSourceId' for connecting to source material.
    - Tags questions as 'Handwritten' for subject/concept.
    """
    data = request.get_json(silent=True) or request.form.to_dict()

    # 1. Extract Fields
    user_id = data.get("userId")
    user_name = data.get("userName", "Manual User")
    title = data.get("title")
    description = data.get("description")
    raw_mcqs = data.get("questions", [])
    linked_source_id = data.get("linkedSourceId")  # <--- GET LINKED ID

    if not all([user_id, title, description]) or not isinstance(raw_mcqs, list):
        return jsonify({"error": "userId, title, description, and a list of 'questions' are required"}), 400

    if not raw_mcqs:
        return jsonify({"error": "Question bank must contain at least one question."}), 400

    indexed_mcqs = []

    # 2. Format Questions & Inject "Handwritten" Metadata
    for i, mcq in enumerate(raw_mcqs):
        if 'options' in mcq and isinstance(mcq['options'], dict):
            mcq['options'] = json.dumps(mcq['options'])

        mcq['documentIndex'] = i
        mcq['questionId'] = str(uuid.uuid4())

        # --- LOGIC ADDED HERE ---
        # Force these fields so the analytics engine treats them as manual/handwritten
        mcq['predicted_subject'] = {
            "label": "Handwritten",
            "confidence": 1.0
        }
        mcq['predicted_concept'] = {
            "label": "Handwritten",
            "confidence": 1.0
        }

        # Tag the individual question with the source ID too (useful for granular filtering)
        if linked_source_id:
            mcq['linkedSourceId'] = linked_source_id
        # ------------------------

        indexed_mcqs.append(mcq)

    # 3. Store Metadata and Questions
    try:
        # We pass linked_source_id to the storage function
        generated_qa_id = store_mcqs_for_manual_creation(
            user_id,
            user_name,
            title,
            description,
            indexed_mcqs,
            linked_source_id=linked_source_id
        )
    except Exception as e:
        print(f"Error storing manual question bank: {e}")
        return jsonify({"error": "Failed to create and store question bank"}), 500

    return jsonify({
        "message": "Question bank created successfully",
        "generatedQAId": generated_qa_id,
        "userId": user_id,
        "userName": user_name,
        "linkedSourceId": linked_source_id,
        "questions_count": len(indexed_mcqs)
    }), 201

# @app.route("/questionId/solution", methods=["POST"])
# def answer_validator(questionId):
#     """
#     API to check validate answers
#     """
#     if not questionId:
#        return jsonify({"error":"questioId is required"}), 400




@app.route("/question-banks/<generatedQAId>", methods=["DELETE"])
def delete_question_bank(generatedQAId):
    """
    API to delete an entire question bank (metadata and all associated questions).
    """
    if not generatedQAId:
        return jsonify({"error": "generatedQAId is required"}), 400

    # Assume this function handles the deletion from both the main
    # and the questions collection using the generatedQAId.
    success = delete_mcq_bank(generatedQAId)

    if success:
        return jsonify({
            "message": f"Question bank '{generatedQAId}' and all associated questions deleted successfully."
        }), 200
    else:
        # Return 404 if the bank wasn't found to delete, or 500 on database error
        return jsonify({
            "error": f"Failed to delete question bank '{generatedQAId}'. It may not exist."
        }), 200


@app.route("/tests/submitted/<testId>", methods=["DELETE"])
def delete_submitted_test(testId):
    """
    API to delete a specific submitted test session result by its ID.
    """
    if not testId:
        return jsonify({"error": "testId is required"}), 400

    success = delete_submitted_test_by_id(testId)

    if success:
        return jsonify({
            "message": f"Submitted test result '{testId}' deleted successfully."
        }), 200
    else:
        return jsonify({
            "error": f"Failed to delete submitted test result '{testId}'. It may not exist."
        }), 404


@app.route("/paper-sets/<testId>", methods=["DELETE"])
def delete_test_session(testId):
    """
    API to delete a specific test session by its ID.
    """
    if not testId:
        return jsonify({"error": "testId is required"}), 400

    # Assume this function handles the deletion from test_sessions_collection
    success = delete_test_session_by_id(testId)

    if success:
        return jsonify({
            "message": f"Test '{testId}' deleted successfully."
        }), 200
    else:
        return jsonify({
            "message": f"Failed to delete '{testId}' "
        }), 200


@app.route("/test-attempts/<attemptId>", methods=["DELETE"])
def delete_submitted_test_attempt_api(attemptId):
    """
    API to delete a specific submitted test attempt by attemptId.
    """
    if not attemptId:
        return jsonify({"error": "attemptId is required"}), 400

    success = delete_submitted_test_attempt(attemptId)
    if not success:
        return jsonify({"error": "Failed to delete attempt"}), 200

    return jsonify({
        "message": f"Attempt {attemptId} deleted successfully"
    }), 200


@app.route("/paper-sets/<testId>", methods=["PUT"])
def edit_paperset(testId):
    """
    Update specific fields of a test session.
    Allows partial updates for test metadata and individual questions.
    """
    payload = request.get_json(silent=True) or {}

    if not testId:
        return jsonify({"error": "testId is required"}), 400

    # 1️⃣ Fetch existing test session
    existing_record = fetch_test_by_testId(testId)
    if not existing_record:
        return jsonify({"error": f"Test session '{testId}' not found"}), 404

    updated_data = existing_record.copy()

    # Extract fields
    edits = payload.get("edits", [])
    new_title = payload.get("testTitle")
    new_total_time = payload.get("totalTime")

    # --- Step 2: Update Top-Level Fields ---
    if new_title is not None:
        updated_data["testTitle"] = new_title

    if new_total_time is not None:
        updated_data["totalTime"] = new_total_time

    # --- Step 3: Question Operations ---
    existing_questions = {q["questionId"]: q for q in updated_data.get("questions", [])}

    for edit in edits:
        operation = edit.get("operation")
        data = edit.get("data")

        if not operation or not data:
            continue

        # ---------- ADD ----------
        if operation == "add":
            qid = data.get("questionId")
            if not qid:
                continue

            # Set default fields for new question
            data.setdefault("documentIndex", len(existing_questions))
            data.setdefault("testIndex", len(existing_questions) + 1)
            data.setdefault("userId", updated_data.get("userId"))
            data.setdefault("generatedQAId", updated_data.get("generatedQAId"))
            data.setdefault("passage", "")
            data.setdefault("image", None)
            data.setdefault("noise", "")

            existing_questions[qid] = data

        # ---------- EDIT ----------
        elif operation == "edit":
            qid = data.get("questionId")
            if qid and qid in existing_questions:
                for key, value in data.items():
                    existing_questions[qid][key] = value

        # ---------- DELETE ----------
        elif operation == "delete":
            qid = data.get("questionId")
            if qid in existing_questions:
                del existing_questions[qid]

    # Sort after update
    updated_data["questions"] = sorted(
        list(existing_questions.values()),
        key=lambda q: q.get("documentIndex", 999999)
    )

    # --- Step 4: Save back ---
    success = update_test_session(testId, updated_data)

    if success:
        return jsonify({
            "message": "Test session updated successfully",
            "testId": testId,
            "updated_fields": list(payload.keys())
        }), 200
    else:
        return jsonify({"error": "Failed to update test session"}), 500


@app.route("/tests/grade-analysis", methods=["POST"])
def grade_test_analysis():
    """
    API to provide AI-powered grading and analysis for submitted answers.
    Payload: { "answers": [ { "questionId": "...", "your_answer": "..." }, ... ] }
    """
    data = request.get_json(silent=True) or {}
    answers = data.get("answers")

    if not answers or not isinstance(answers, list):
        return jsonify({"error": "A list of answers with questionId is required"}), 400

    analysis_results = []

    for user_ans_obj in answers:
        question_id = user_ans_obj.get("questionId")
        student_answer = user_ans_obj.get("your_answer", "")

        if not question_id:
            continue

        # 1. Fetch context (Knowledge Base, Question, Passage) from Vector DB
        # This function retrieves 'question', 'passage', and 'knowledge_base'
        context = fetch_question_context(question_id)

        if not context:
            analysis_results.append({
                "questionId": question_id,
                "status": "error",
                "message": "Question context not found in database"
            })
            continue

        # 2. Combine Passage and Knowledge Base for the AI
        combined_kb = f"Passage: {context['passage']}\n\nSource Info: {context['knowledge_base']}"
        question_text = context['question']

        # 3. Call the Gradio Grading API
        try:
            # get_grading_report sends data to the 'heerjtdev/answer_validator' API
            report = get_grading_report(
                kb_text=combined_kb,
                question_text=question_text,
                answer_text=student_answer
            )

            # --- DEBUG PRINT: View the raw Gradio response in your console ---
            print(f"\n[DEBUG] Question ID: {question_id}")
            print(f"[DEBUG] Raw report from get_grading_report: {report}")

            analysis_results.append({
                "questionId": question_id,
                "question": question_text,
                "student_answer": student_answer,
                "analysis": report
            })
        except Exception as e:
            print(f"[ERROR] Grading service failed for {question_id}: {e}")
            analysis_results.append({
                "questionId": question_id,
                "status": "error",
                "message": f"Grading service failed: {str(e)}"
            })

    return jsonify({
        "status": "success",
        "results": analysis_results
    }), 200




@app.route("/tests/grade-descriptive/<attemptId>", methods=["POST"])
def grade_descriptive_questions(attemptId):
    """
    Grade all pending descriptive questions for a specific test attempt.
    Calculates partial credit based on suggested_mark and updates aggregate scores.
    """
    try:
        # Fetch the submission
        results = client.retrieve(
            collection_name=COLLECTION_SUBMITTED,
            ids=[attemptId],
            with_payload=True,
            with_vectors=False
        )

        if not results:
            return jsonify({"error": "Submission not found"}), 404

        payload = _extract_payload(results[0])
        detailed_results = json.loads(payload.get("detailed_results", "[]"))

        # Map to store full AI feedback reports
        ai_feedback = {}
        total_descriptive_points = 0.0
        graded_count = 0

        # Process each descriptive question
        for result in detailed_results:
            if result.get("question_type") == "DESCRIPTIVE" and result.get("status") == "pending_grading":
                question_id = result.get("questionId")
                student_answer = result.get("your_answer", "")

                # Fetch context for grading
                context = fetch_question_context(question_id)
                if not context:
                    result["status"] = "grading_error"
                    result["error"] = "Context not found"
                    continue

                combined_kb = f"Passage: {context['passage']}\n\nSource Info: {context['knowledge_base']}"
                question_text = context['question']

                try:
                    # Call grading API
                    report = get_grading_report(
                        kb_text=combined_kb,
                        question_text=question_text,
                        answer_text=student_answer
                    )

                    # Extract score (0-10) and calculate partial credit (0.0-1.0)
                    # Support both 'total_score' and 'suggested_mark' keys found in your logs
                    ai_score = float(report.get("suggested_mark") or report.get("total_score") or 0)

                    # Store the FULL AI feedback report
                    ai_feedback[question_id] = report

                    # Update the result with partial credit info
                    partial_credit = ai_score / 10.0
                    result["ai_score"] = ai_score
                    result["partial_score"] = partial_credit
                    result["status"] = "graded"

                    # For UI icons: consider > 50% as 'correct'
                    result["is_correct"] = ai_score >= 5.0

                    total_descriptive_points += partial_credit
                    graded_count += 1

                except Exception as e:
                    print(f"[ERROR] Grading failed for {question_id}: {e}")
                    result["status"] = "grading_error"
                    result["error"] = str(e)

        # Recalculate aggregate totals
        total_questions = int(payload.get("total_questions", 0))
        mcq_correct = float(payload.get("mcq_correct", 0))

        # Sum of MCQ points and Partial Descriptive points
        total_correct = mcq_correct + total_descriptive_points
        final_score = round((total_correct / total_questions) * 100, 2) if total_questions > 0 else 0.0

        # Update the submission in Qdrant with floating point results
        updated_payload = payload.copy()
        updated_payload["detailed_results"] = json.dumps(detailed_results)
        updated_payload["total_correct"] = total_correct  # Now a float (e.g., 0.5)
        updated_payload["score"] = final_score
        updated_payload["grading_status"] = "complete"
        updated_payload["ai_feedback"] = json.dumps(ai_feedback)

        # Use set_payload to preserve existing vectors while updating data
        client.set_payload(
            collection_name=COLLECTION_SUBMITTED,
            payload=updated_payload,
            points=[attemptId]
        )

        return jsonify({
            "status": "success",
            "attemptId": attemptId,
            "final_score": final_score,
            "total_correct": total_correct,
            "grading_status": "complete",
            "ai_feedback": ai_feedback,
            "detailed_results": detailed_results
        })

    except Exception as e:
        print(f"[ERROR] grade_descriptive_questions: {e}")
        return jsonify({"error": str(e)}), 500




@app.route("/admin/subscriptions/assign", methods=["POST"])
def assign_subscription():
    data = request.json
    admin_id = data.get("adminId")  # Use this to check permissions
    target_user_id = data.get("userId")
    qbank_id = data.get("generatedQAId")

    # TODO: Add your logic here to check if admin_id has 'special permissions'
    # if not is_admin(admin_id): return jsonify({"error": "Unauthorized"}), 403

    try:
        sub_id = add_subscription_record(target_user_id, qbank_id)
        return jsonify({"message": "Assigned successfully", "subscriptionId": sub_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/user/subscriptions/questions", methods=["GET"])
def get_my_subscribed_content():
    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "userId required"}), 400

    questions = fetch_subscribed_questions(user_id)
    return jsonify({
        "count": len(questions),
        "questions": questions
    })





@app.route("/marketplace/download", methods=["POST"])
def download_curated_bank():
    data = request.json
    user_id = data.get("userId")
    qbank_id = data.get("generatedQAId")

    if not user_id or not qbank_id:
        return jsonify({"error": "Missing ID"}), 400

    try:
        # This adds the 'Subscription' record
        sub_id = add_subscription_record(user_id, qbank_id)
        return jsonify({"message": "Download successful", "id": sub_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/question-banks/<generatedQAId>/publish", methods=["PUT"])
def publish_to_marketplace(generatedQAId):
    data = request.json or {}
    admin_id = data.get("adminId")
    is_public = data.get("isPublic", True)

    if not is_admin(admin_id):
        return jsonify({"error": "Unauthorized"}), 403

    success = toggle_bank_public_status(generatedQAId, is_public)

    if success:
        return jsonify({
            "message": "Published to marketplace" if is_public else "Unpublished",
            "isPublic": is_public
        }), 200
    else:
        return jsonify({"error": "Failed to update"}), 500


# API to view marketplace
@app.route("/marketplace", methods=["GET"])
def get_marketplace():
    banks = fetch_public_marketplace()
    return jsonify(banks), 200







import threading

@app.route("/user/update-username", methods=["POST"])
def api_update_username():
    data = request.json
    user_id = data.get("userId")
    new_username = data.get("userName")

    if not user_id or not new_username:
        return jsonify({"error": "Missing data"}), 400

    # Start the heavy Qdrant update in the BACKGROUND
    # This allows the API to return 200 OK immediately
    thread = threading.Thread(
        target=update_user_metadata_in_qdrant,
        args=(user_id, new_username)
    )
    thread.start()

    return jsonify({
        "message": "Update started in background",
        "status": "processing"
    }), 200






@app.route("/marketplace/community", methods=["GET"])
def get_community_marketplace():
    """API to view user-generated (non-admin) public banks."""
    try:
        # Calls the helper function from vector_db.py
        banks = fetch_community_marketplace(limit=20)
        return jsonify(banks), 200
    except Exception as e:
        print(f"Error fetching community marketplace: {e}")
        return jsonify({"error": str(e)}), 500






@app.route("/question-banks/init", methods=["POST"])
def init_question_bank():
    """
    Initializes a new question bank with default 'Untitled' metadata.
    Returns the generatedQAId immediately for the frontend editor.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    user_id = data.get("userId")

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    # Use defaults as requested
    default_title = "Untitled"
    default_desc = "write a description"

    try:
        generated_qa_id = initialize_bank_record(
            userId=user_id,
            title=default_title,
            description=default_desc
        )

        if not generated_qa_id:
            return jsonify({"error": "Failed to initialize bank record"}), 500

    except Exception as e:
        print(f"Error initializing question bank: {e}")
        return jsonify({"error": "Internal server error"}), 500

    return jsonify({
        "message": "Question bank initialized successfully",
        "generatedQAId": generated_qa_id,
        "userId": user_id,
        "title": default_title,
        "description": default_desc,
        "questions_count": 0
    }), 201





@app.route("/flashcards/init", methods=["POST"])
def init_flashcard_deck():
    data = request.get_json(silent=True) or request.form.to_dict()
    user_id = data.get("userId")
    title = data.get("title", "New Flashcard Deck")
    description = data.get("description", "No description provided")

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    try:
        # Pass 'FLASHCARD' as the record_type
        generated_qa_id = initialize_bank_record(
            userId=user_id,
            title=title,
            description=description,
            record_type="FLASHCARD"
        )

        if not generated_qa_id:
            return jsonify({"error": "Failed to initialize deck"}), 500

    except Exception as e:
        print(f"Error initializing flashcard deck: {e}")
        return jsonify({"error": "Internal server error"}), 500

    return jsonify({
        "message": "Flashcard deck created successfully",
        "generatedQAId": generated_qa_id,
        "type": "FLASHCARD"
    }), 201






@app.route("/flashcards/list", methods=["GET"])
def get_user_flashcards():
    user_id = request.args.get("userId")

    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    try:
        # Call the helper function
        decks = fetch_user_flashcards(user_id)

        return jsonify({"decks": decks}), 200

    except Exception as e:
        # Log the error (optional)
        return jsonify({"error": "Failed to fetch flashcards", "details": str(e)}), 500




# Import the helper function we created earlier
# Assuming it is in a file named 'grading_helper.py' or defined above
# from grading_helper import grade_student_answer

@app.route("/grade-single-answer", methods=["POST"])
def grade_single_answer():
    """
    Stateless grading endpoint.
    Accepts question, answer, and context. Returns AI feedback immediately.
    Does NOT store data in Qdrant/Database.
    """
    try:
        data = request.get_json()

        # 1. Extract necessary fields from the frontend request
        question = data.get("question")
        student_answer = data.get("student_answer")
        context = data.get("context")
        max_marks = data.get("max_marks", 5)

        # 2. Basic Validation
        if not question or not student_answer:
            return jsonify({"error": "Missing 'question' or 'student_answer'"}), 400

        if not context:
            return jsonify({"error": "Missing 'context'. Grading requires reference material."}), 400

        # 3. Call the helper function (The one using Gradio Client)
        # This function handles the connection to the Hugging Face Space
        ai_result = grade_student_answer(
            question=question,
            student_answer=student_answer,
            context_text=context,
            max_marks=max_marks
        )

        # 4. Check for internal errors in the helper
        if ai_result.get("error"):
            return jsonify({"status": "error", "message": ai_result["error"]}), 500

        # 5. Return the result directly to the frontend
        # The frontend can now decide whether to show it, save it to local storage, etc.
        return jsonify({
            "status": "success",
            "question": question,
            "student_answer": student_answer,
            "ai_feedback": ai_result["grading_feedback"],
            "evidence_used": ai_result["evidence_used"],
            # If you need to parse the score out specifically:
            "raw_result": ai_result
        })

    except Exception as e:
        print(f"[ERROR] /grade-single-answer: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/sources/upload", methods=["POST"])
def upload_source_material_endpoint():
    """
    Uploads raw text to be used as reference material (RAG).
    Expects 'text' string from frontend (which handles PDF->Text conversion).
    Automatically chunks the text before storing.
    """
    print(f"\n[START] /sources/upload request received")

    # 1. Validate Inputs (Accept JSON or Form Data)
    data = request.get_json(silent=True) or request.form

    user_id = data.get("userId")
    title = data.get("title")
    raw_text = data.get("text")

    if not user_id or not raw_text:
        return jsonify({"error": "userId and text are required"}), 400

    if not title:
        title = "Untitled Source"

    try:
        # 2. Chunk the text
        # Since we receive one big string, we must split it.
        # Embedding models usually have a limit (e.g. 512 tokens), so we split by ~1000 chars.
        print(f"[STEP] Chunking text for {title}...")

        chunk_size = 1000
        overlap = 100
        text_chunks_list = []

        # Simple sliding window chunking
        start = 0
        page_counter = 1

        while start < len(raw_text):
            end = start + chunk_size
            chunk = raw_text[start:end]

            # Formatting for store_source_material
            text_chunks_list.append({
                "text": chunk,
                "page": page_counter
            })

            # Move forward, keeping some overlap to maintain context
            start += (chunk_size - overlap)
            page_counter += 1

        if not text_chunks_list:
            return jsonify({"error": "Text provided was empty."}), 400

        # 3. Store in Qdrant via vector_db
        print(f"[STEP] Storing {len(text_chunks_list)} chunks in vector DB...")
        source_id = store_source_material(user_id, title, text_chunks_list)

        if source_id:
            return jsonify({
                "message": "Source material uploaded successfully",
                "sourceId": source_id,
                "chunks_processed": len(text_chunks_list),
                "title": title
            }), 201
        else:
            return jsonify({"error": "Failed to store source material"}), 500

    except Exception as e:
        print(f"[ERROR] upload_source_material_endpoint: {e}")
        return jsonify({"error": str(e)}), 500






@app.route("/sources/list", methods=["GET"])
def list_user_sources():
    """Fetches the list of uploaded source PDFs for a user."""
    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({"error": "userId is required"}), 400

    sources = fetch_user_sources(user_id)
    return jsonify(sources), 200


@app.route("/sources/<sourceId>", methods=["DELETE"])
def delete_source_endpoint(sourceId):
    """Deletes a source file and its associated text chunks."""
    if not sourceId:
        return jsonify({"error": "sourceId is required"}), 400

    success = delete_source_material(sourceId)
    if success:
        return jsonify({"message": "Source deleted successfully"}), 200
    else:
        return jsonify({"error": "Failed to delete source"}), 500


@app.route("/sources/<sourceId>/content", methods=["GET"])
def get_source_content(sourceId):
    """
    Fetches the full text content of a specific source file.
    """
    if not sourceId:
        return jsonify({"error": "sourceId is required"}), 400

    # 1. Fetch text from Vector DB
    full_text = fetch_full_source_text(sourceId)

    if full_text is None:
        # It might be None if ID is wrong or DB error,
        # but also check if the metadata exists to be sure it's a 404 vs empty file
        return jsonify({"error": "Source not found or failed to load"}), 404

    return jsonify({
        "sourceId": sourceId,
        "content": full_text
    }), 200



@app.route("/htr", methods=["POST"])
def handwritten_text_recognition():
    """
    Receives an image file, sends it to the LastStraw Space,
    and returns the extracted text without saving it to the DB.
    """
    # 1. Check if image part exists
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    # 2. Check if filename is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_path = None
    try:
        # 3. Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        # 4. Call the helper function
        extracted_text = extract_text_from_image(temp_path)

        if extracted_text is None:
             return jsonify({"error": "Failed to extract text from image"}), 500

        # 5. Return result
        return jsonify({
            "message": "Text extracted successfully",
            "text": extracted_text
        }), 200

    except Exception as e:
        print(f"[ERROR] /htr: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # 6. Cleanup: Remove the temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
