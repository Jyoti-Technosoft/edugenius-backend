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
from gradio_api import call_yolo_api,latex_model, call_feeedback_api, get_grading_report

"""
===========================================================


MODEL OPTIONS


===========================================================
"""
app = Flask(__name__)
CORS(app)

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
    fetch_submitted_test_by_testId, delete_submitted_test_attempt, update_answer_flag_in_qdrant, normalize_answer,fetch_question_banks_metadata, fetch_question_context, client, COLLECTION_SUBMITTED, embed, _extract_payload


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

@app.route("/question-banks/upload", methods=["POST"])
def upload_pdf():
    print(f"\n[START] /create_question_bank request received")

    # 1. Validate inputs
    user_id = request.form.get("userId")
    title = request.form.get("title")
    description = request.form.get("description")
    pdf_file = request.files.get("pdf")

    print(f"[INFO] Received form-data: userId={user_id}, title={title}, description={description}")
    if not pdf_file:
        return jsonify({"error": "PDF file not provided"}), 400

    if not all([user_id, title, description]):
        return jsonify({"error": "userId, title, description are required"}), 400

    # 2. Keep PDF in memory (no Drive)
    print("[STEP] Reading PDF into memory...")
    pdf_bytes = pdf_file.read()
    pdf_name = secure_filename(pdf_file.filename)

    # 3. Directly call model
    print("[STEP] Calling LayoutLM model directly (no Drive)...")
    # final_data = call_layoutlm_api(pdf_bytes, pdf_name)
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
            "questionId": str(uuid.uuid4())  # ✅ assign unique ID
        }
        for i, mcq in enumerate(final_data)
    ]

    # 5. Store in vector DB
    print("[STEP] Storing Question Bank in vector database...")
    createdAtTimestamp = datetime.now().isoformat()
    stored_id, all_have_answers = store_mcqs(
        user_id, title, description, indexed_mcqs, pdf_name, createdAtTimestamp
    )
    print(f"[SUCCESS] Stored with generatedQAId={stored_id}")

    print("[END] Request complete\n")
    return Response(
        json.dumps({
            "generatedQAId": stored_id,
            "userId": user_id,
            "fileName": pdf_name,
            "createdAt": createdAtTimestamp,
            "answerFound": all_have_answers
        }, ensure_ascii=False),
        mimetype="application/json"
    )



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



@app.route("/question-banks/upload/images", methods=["POST"])
def upload_image():
    print("\n[START] /create_question_bank request received")

    # 1. Validate inputs
    user_id = request.form.get("userId")
    title = request.form.get("title")
    description = request.form.get("description")
    image_files = request.files.getlist("image")  # ✅ multiple images

    print(f"[INFO] Received form-data: userId={user_id}, title={title}, description={description}")
    if not image_files or len(image_files) == 0:
        return jsonify({"error": "No image file(s) provided"}), 400

    if not all([user_id, title, description]):
        return jsonify({"error": "userId, title, description are required"}), 400

    all_results = []

    # 2. Loop through each image
    for idx, img_file in enumerate(image_files, start=1):
        print(f"[STEP] Reading image {idx}/{len(image_files)} into memory...")
        file_bytes = img_file.read()
        filename = secure_filename(img_file.filename)

        # 3. Directly call model for each image
        print(f"[STEP] Calling LayoutLM model for {filename} ...")
        try:
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
        {**mcq, "documentIndex": i}
        for i, mcq in enumerate(all_results)
    ]

    # 5. Store in vector DB
    print("[STEP] Storing Question Bank in vector database...")
    createdAtTimestamp = datetime.now().isoformat()
    stored_id = store_mcqs(
        user_id, title, description, indexed_mcqs, "multiple_images.zip", createdAtTimestamp
    )
    print(f"[SUCCESS] Stored with generatedQAId={stored_id}")

    print("[END] Request complete\n")
    return Response(
        json.dumps({
            "generatedQAId": stored_id,
            "userId": user_id,
            "fileCount": len(image_files),
            "createdAt": createdAtTimestamp,
        }, ensure_ascii=False),
        mimetype="application/json"
    )


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
    limit = request.args.get('limit', default=10, type=int)

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
#
# @app.route("/tests/submit", methods=["POST"])
# def submit_test():
#     """
#     API to submit student answers, check correctness,
#     calculate score, and store submission data.
#     Frontend sends: userId, testId, testTitle, timeSpent, totalTime, answers[]
#     """
#     data = request.get_json(silent=True) or {}
#
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
#     # 🧠 Fetch original test data (includes correct answers)
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
#     # Build quick lookup of correct answers
#     correct_map = {q.get("questionId"): q.get("answer") for q in questions}
#
#     totalQuestions = len(correct_map)
#     total_correct = 0
#     results = []
#
#     # ✅ Compare each submitted answer
#     for ans in answers:
#         qid = ans.get("questionId")
#         qtext = ans.get("question")
#         user_ans = ans.get("your_answer")
#
#         # Try to get correct answer using questionId first, then question text
#         correct_ans = None
#         if qid and qid in correct_map:
#             correct_ans = correct_map.get(qid)
#         elif qtext:
#             for q in questions:
#                 if qtext.strip().lower() == q.get("question", "").strip().lower():
#                     correct_ans = q.get("answer")
#                     qid = q.get("questionId")
#                     break
#
#         is_correct = (normalize_answer(user_ans) == normalize_answer(correct_ans))
#
#         if is_correct:
#             total_correct += 1
#
#         results.append(OrderedDict([
#             ("questionId", qid),
#             ("your_answer", user_ans),
#             ("correct_answer", correct_ans),
#             ("is_correct", is_correct)
#         ]))
#
#     # 🧮 Calculate score
#     score = round((total_correct / totalQuestions) * 100, 2) if totalQuestions > 0 else 0.0
#
#     # 💾 Store submission attempt in Qdrant or DB
#     is_stored, attemptId = store_submitted_test(
#         userId=userId,
#         testId=testId,
#         testTitle=testTitle,
#         timeSpent=timeSpent,
#         totalTime=totalTime,
#         submittedAt=submittedAt,
#         detailed_results=results,
#         score=score,
#         total_questions=totalQuestions,
#         total_correct=total_correct
#     )
#
#     if not is_stored:
#         return jsonify({"error": "Failed to store submission"}), 500
#
#     # 📦 Final response
#     response = OrderedDict([
#         ("attemptId", attemptId),
#         ("userId", userId),
#         ("testId", testId),
#         ("testTitle", testTitle),
#         ("submittedAt", submittedAt),
#         ("timeSpent", timeSpent),
#         ("total_questions", totalQuestions),
#         ("total_correct", total_correct),
#         ("score", score),
#         ("detailed_results", results)
#     ])
#
#     return jsonify(response)


@app.route("/tests/submit", methods=["POST"])
def submit_test():
    """
    API to submit student answers, check correctness for MCQs,
    defer grading for Descriptive questions, calculate score, and store submission data.
    Frontend sends: userId, testId, testTitle, timeSpent, totalTime, answers[]
    """
    data = request.get_json(silent=True) or {}
    userId = data.get("userId")
    testId = data.get("testId")
    testTitle = data.get("testTitle")
    timeSpent = data.get("timeSpent")
    totalTime = data.get("totalTime")
    answers = data.get("answers")

    if not all([userId, testId, answers]):
        return jsonify({"error": "Missing required fields: userId, testId, answers"}), 400
    if not isinstance(answers, list):
        return jsonify({"error": "Answers must be a list"}), 400

    submittedAt = datetime.now().isoformat()

    # 🧠 Fetch original test data
    test_data = fetch_test_by_testId(testId)
    if not test_data:
        return jsonify({"error": "Test not found"}), 404

    questions = test_data.get("questions", [])
    if isinstance(questions, str):
        try:
            questions = json.loads(questions)
        except Exception:
            questions = []

    # Build lookup for questions
    question_map = {}
    for q in questions:
        qid = q.get("questionId")
        if qid:
            question_map[qid] = q

    totalQuestions = len(question_map)
    total_correct = 0
    total_mcq = 0
    total_descriptive = 0
    mcq_correct = 0
    results = []
    descriptive_question_ids = []

    # ✅ Process each submitted answer based on question type
    for ans in answers:
        qid = ans.get("questionId")
        qtext = ans.get("question")
        user_ans = ans.get("your_answer")

        # Find the question details
        question_details = None
        if qid and qid in question_map:
            question_details = question_map.get(qid)
        elif qtext:
            for q in questions:
                if qtext.strip().lower() == q.get("question", "").strip().lower():
                    question_details = q
                    qid = q.get("questionId")
                    break

        if not question_details:
            results.append(OrderedDict([
                ("questionId", qid),
                ("question", qtext),
                ("question_type", "unknown"),
                ("your_answer", user_ans),
                ("correct_answer", None),
                ("is_correct", False),
                ("status", "question_not_found")
            ]))
            continue

        question_type = question_details.get("question_type", "MCQ").upper()

        if question_type == "MCQ":
            # ✅ MCQ: Check answer immediately
            total_mcq += 1
            correct_ans = question_details.get("answer")
            is_correct = (normalize_answer(user_ans) == normalize_answer(correct_ans))

            if is_correct:
                total_correct += 1
                mcq_correct += 1

            results.append(OrderedDict([
                ("questionId", qid),
                ("question", question_details.get("question", "")),
                ("question_type", "MCQ"),
                ("your_answer", user_ans),
                ("correct_answer", correct_ans),
                ("is_correct", is_correct),
                ("status", "graded")
            ]))

        else:  # DESCRIPTIVE
            # ⏳ Descriptive: Store answer, defer grading
            total_descriptive += 1
            descriptive_question_ids.append(qid)
            knowledge_base = question_details.get("knowledge_base", "")

            results.append(OrderedDict([
                ("questionId", qid),
                ("question", question_details.get("question", "")),
                ("question_type", "DESCRIPTIVE"),
                ("your_answer", user_ans),
                ("correct_answer", knowledge_base),
                ("is_correct", None),
                ("ai_score", None),
                ("status", "pending_grading")
            ]))

    # 🧮 Calculate preliminary score (only from MCQs)
    if totalQuestions > 0:
        preliminary_score = round((total_correct / totalQuestions) * 100, 2)
    else:
        preliminary_score = 0.0

    # 💾 Store submission attempt
    is_stored, attemptId = store_submitted_test(
        userId=userId,
        testId=testId,
        testTitle=testTitle,
        timeSpent=timeSpent,
        totalTime=totalTime,
        submittedAt=submittedAt,
        detailed_results=results,
        score=preliminary_score,
        total_questions=totalQuestions,
        total_correct=total_correct,
        total_mcq=total_mcq,
        total_descriptive=total_descriptive,
        mcq_correct=mcq_correct,
        grading_status="partial" if total_descriptive > 0 else "complete",
        ai_feedback=None  # Will be filled by grade_descriptive endpoint
    )

    if not is_stored:
        return jsonify({"error": "Failed to store submission"}), 500

    # 📦 Final response
    response = OrderedDict([
        ("attemptId", attemptId),
        ("userId", userId),
        ("testId", testId),
        ("testTitle", testTitle),
        ("submittedAt", submittedAt),
        ("timeSpent", timeSpent),
        ("total_questions", totalQuestions),
        ("total_mcq", total_mcq),
        ("total_descriptive", total_descriptive),
        ("mcq_correct", mcq_correct),
        ("total_correct", total_correct),
        ("score", preliminary_score),
        ("grading_status", "partial" if total_descriptive > 0 else "complete"),
        ("descriptive_questions_pending", descriptive_question_ids),
        ("detailed_results", results)
    ])

    return jsonify(response)



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
    else:
        edits = payload.get("edits")
        new_title = payload.get("title")
        new_description = payload.get("description")

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
                description=new_description
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


@app.route("/question-banks/manual", methods=["POST"])
def create_manual_question_bank():
    """
    API to create a new question bank and populate it with a list of questions
    in a single request for a smoother user experience.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    user_id = data.get("userId")
    title = data.get("title")
    description = data.get("description")
    raw_mcqs = data.get("questions", [])  # Expects a list of question objects

    if not all([user_id, title, description]) or not isinstance(raw_mcqs, list):
        return jsonify({"error": "userId, title, description, and a list of 'questions' are required"}), 400

    if not raw_mcqs:
        return jsonify({"error": "Question bank must contain at least one question."}), 400

    indexed_mcqs = []

    # 1. Format and Index MCQs (similar to your upload_pdf route logic)
    for i, mcq in enumerate(raw_mcqs):
        # Ensure options are properly formatted (if they come as a dict from the client)
        if 'options' in mcq and isinstance(mcq['options'], dict):
            # We need to ensure the options are stored as a JSON string
            # as required by the ChromaDB metadata constraint (as discovered earlier).
            mcq['options'] = json.dumps(mcq['options'])

        # NOTE: If your database requires questionId/documentIndex, they must be set here.
        # However, we will assume 'store_mcqs_for_manual_creation' handles questionId and documentIndex assignment.
        mcq['documentIndex'] = i
        mcq['questionId'] = str(uuid.uuid4())
        indexed_mcqs.append(mcq)

    # 2. Store Metadata and Questions (using a modified store function)
    try:
        # Create a function similar to store_mcqs but for manual data
        generated_qa_id = store_mcqs_for_manual_creation(
            user_id,
            title,
            description,
            indexed_mcqs
        )
    except Exception as e:
        print(f"Error storing manual question bank: {e}")
        return jsonify({"error": "Failed to create and store question bank"}), 500

    return jsonify({
        "message": "Question bank created and populated successfully",
        "generatedQAId": generated_qa_id,
        "userId": user_id,
        "title": title,
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
    Stores the full AI feedback report for each question.
    Frontend should call this immediately after /tests/submit if there are descriptive questions.
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

        # This will store the full AI feedback for each descriptive question
        ai_feedback = {}

        total_descriptive_score = 0
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

                # Prepare data for AI grading
                combined_kb = f"Passage: {context['passage']}\n\nSource Info: {context['knowledge_base']}"
                question_text = context['question']

                try:
                    # Call grading API - this returns the full report
                    report = get_grading_report(
                        kb_text=combined_kb,
                        question_text=question_text,
                        answer_text=student_answer
                    )

                    print(f"\n[DEBUG] AI Grading Report for {question_id}:")
                    print(json.dumps(report, indent=2))

                    # Extract score from the report
                    ai_score = float(report.get("total_score", 0))

                    # Store the FULL AI feedback report
                    ai_feedback[question_id] = report

                    # Update the result with AI grading info
                    result["ai_score"] = ai_score
                    result["status"] = "graded"

                    # Determine if answer is correct based on score threshold
                    # You can adjust this threshold (currently 6 out of 10)
                    score_threshold = 6.0
                    result["is_correct"] = ai_score >= score_threshold

                    total_descriptive_score += ai_score
                    graded_count += 1

                except Exception as e:
                    print(f"[ERROR] Grading failed for {question_id}: {e}")
                    result["status"] = "grading_error"
                    result["error"] = str(e)

        # Recalculate total score
        total_questions = int(payload.get("total_questions", 0))
        mcq_correct = int(payload.get("mcq_correct", 0))

        # Count descriptive questions considered correct
        descriptive_correct = sum(1 for r in detailed_results
                                  if r.get("question_type") == "DESCRIPTIVE"
                                  and r.get("is_correct") == True)

        total_correct = mcq_correct + descriptive_correct
        final_score = round((total_correct / total_questions) * 100, 2) if total_questions > 0 else 0.0

        # Calculate average descriptive score (out of 10)
        descriptive_average = round(total_descriptive_score / graded_count, 2) if graded_count > 0 else 0

        # Update the submission with complete grading
        updated_payload = payload.copy()
        updated_payload["detailed_results"] = json.dumps(detailed_results)
        updated_payload["total_correct"] = total_correct
        updated_payload["score"] = final_score
        updated_payload["grading_status"] = "complete"
        updated_payload["descriptive_average_score"] = descriptive_average
        updated_payload["ai_feedback"] = json.dumps(ai_feedback)  # 🟢 Store full AI feedback

        vec = embed(f"submitted_test:{payload['testId']}:{attemptId}")[0]
        p = models.PointStruct(id=attemptId, vector=vec, payload=updated_payload)
        client.upsert(collection_name=COLLECTION_SUBMITTED, points=[p])

        return jsonify({
            "status": "success",
            "attemptId": attemptId,
            "final_score": final_score,
            "total_correct": total_correct,
            "grading_status": "complete",
            "descriptive_average_score": descriptive_average,
            "graded_count": graded_count,
            "ai_feedback": ai_feedback,  # Return full feedback to frontend
            "detailed_results": detailed_results
        })

    except Exception as e:
        print(f"[ERROR] grade_descriptive_questions: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
