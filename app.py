import uuid
from collections import Counter
import pickle
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import fitz  # PyMuPDF
import pytesseract
import os
import re
import json
from PIL import Image, ImageEnhance
from datetime import datetime
import google.generativeai as genai
import base64
import random
from gradio import process_pdf_pipeline

"""
===========================================================


MODEL OPTIONS


===========================================================
"""


#from working_prototype_layout_safety import process_pdf_pipeline #[UNCOMMENT THIS FOR USING THE FINETUNED LAYOUTLMv3+CRF MODEL]
from working_prototype_layout_LSTM import process_pdf_pipeline  # Bi-LSTM_CRF MODEL
#from BERT_CRF_INFERENCE import process_pdf_pipeline  # BERT_CRF_INFERENCE MODEL

from matplotlib.cbook import CallbackRegistry
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.Model import Model

pytesseract.pytesseract.tesseract_cmd = r"./usr/bin/tesseract"

app = Flask(__name__)
CORS(app)

from collections import OrderedDict

"""
====================================================================

Helper Functions

====================================================================
"""

from vector_db import store_mcqs, fetch_mcqs,fetch_random_mcqs, store_test_session, fetch_test_by_testId,test_sessions_by_userId,store_submitted_test, submitted_tests_by_userId,add_single_question,update_single_question,delete_single_question,store_mcqs_for_manual_creation, delete_mcq_bank, delete_submitted_test_by_id, delete_test_session_by_id, update_test_session, update_question_bank_metadata
from werkzeug.utils import secure_filename

#
# ==================================================
#
# GEMINI API FUNCTION

# uncomment this entire text block and use your API-KEY to use Gemini AI to extract question and answers
#this file is using our pre-trained model by default
#
# ===================================================
 #from examples.transformation.signature_method import model

# genai.configure(api_key="test-key")
# model = genai.GenerativeModel("gemini-2.0-flash")
#

# def encode_image_to_base64(image_path):
#     with open(image_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode("utf-8")
#
# def fix_indic_spacing(text):
#     pattern = r'([\u0900-\u097F\u0A80-\u0AFF])\s+([\u0900-\u097F\u0A80-\u0AFF])'
#     while re.search(pattern, text):
#         text = re.sub(pattern, r'\1\2', text)
#     return text
# #
# # Prompt for Gemini
# def get_prompt(text):
#     return f"""
# You are given raw text or OCR content from a multiple-choice question (MCQ) document.
#
# 🔤 The questions and options may be written in English, Hindi, Gujarati, or other Indian languages.
#
# Some questions may include embedded images marked as:
#     [IMAGE_QUESTION: filename.png]
#
# Each MCQ may appear in one of the following formats:
# - Options labeled as A., B., C., D.
# - a) / b) / c)
# - 1. / 2. / 3. / 4.
# - i. / ii. / iii. / iv.
# - Or inside a table
# - Or the correct answer appears later as:
#   - ANSWER: B
#   - Answer - C
#
# Your task is to extract **every complete MCQ** and convert it into valid JSON like this, i want noise to be classified as well:
#
# [
#   {{
#     "question": "What is bookkeeping?",
#     "image": "page3_img1.png",  # Optional
#     "noise" : "the following question paper is designed for IIT Roorkee" #noise that cannot be classified as either question/answer/options but preceeds that particular question
#     "options": {{
#       "A": "Records only income",
#       "B": "Each entry has two sides",
#       "C": "Tracks expenses only",
#       "D": "Used for banks only"
#     }},
#     "answer": "B"
#   }}
# ]
#
# ✅ Rules:
# - Normalize all option labels
# - Remove newlines inside strings
# - If a question has [IMAGE_QUESTION: xyz.png], extract the image name and include it in the `"image"` field.
# - Return a valid JSON array only
# - Each MCQ must have: question text, 2–4 options, and one correct answer
# - If answer not immediately after the question, then look through the rest of the document for an 'answer key' present either at the end of a page or at the end of a section or the document and fill the answer field accordingly
# - If text is from OCR and words appear stuck together (like ફસ્ર્લેડીકોનીપત્નીનેકહેવામાંઆવેછે),
#   split or normalize them into natural readable words (e.g., "ફર્સ્ટ લેડીની પત્નીને કહેવામાં આવે છે")
# Here is the text:
#
# {text}
# """
#
#
#
# def extract_text_chunks(pdf_path, output_path="raw_text.txt", pages_per_chunk=10, overlap=2):
#     """
#     PDF extractor that saves the extracted text to a file.
#     """
#     image_counter = 1
#     os.makedirs("static", exist_ok=True)
#
#     doc = fitz.open(pdf_path)
#     i = 0
#     full_text = []
#
#     while i < len(doc):
#         chunk_texts = []
#         for j in range(i, min(i + pages_per_chunk, len(doc))):
#             page = doc[j]
#             blocks = []
#             is_mcq_page = False
#
#             # Enhanced text extraction with position tracking
#             try:
#                 page_dict = page.get_text("dict", sort=True)
#                 for block in page_dict.get("blocks", []):
#                     bbox = block.get("bbox", [0, 0, 0, 0])
#                     sort_key = (bbox[1], bbox[0])
#                     if block.get("type") == 0:  # Text block
#                         text_block = ""
#                         for line in block.get("lines", []):
#                             for span in line.get("spans", []):
#                                 text_block += span.get("text", "")
#                             text_block += "\n"
#                         text_block = text_block.strip()
#                         if text_block:
#                             if re.search(r'^[A-D][\.\)]\s', text_block, re.MULTILINE):
#                                 is_mcq_page = True
#                             blocks.append((sort_key, text_block, "text"))
#                     elif block.get("type") == 1:  # Image block
#                         try:
#                             img_bytes = block.get("image")
#                             if img_bytes:
#                                 img_filename = f"img{image_counter}.png"
#                                 img_path = os.path.join("static", img_filename)
#                                 with open(img_path, "wb") as f:
#                                     f.write(img_bytes)
#                                 blocks.append((sort_key, f"[IMAGE: {img_filename}]", "image"))
#                                 image_counter += 1
#                         except Exception as e:
#                             print(f"[Warning] Failed to save image on page {j + 1}: {e}")
#             except Exception as e:
#                 print(f"[Warning] Failed to parse layout on page {j + 1}: {e}")
#
#             # OCR fallback with position mapping
#             if len(" ".join(t[1] for t in blocks if t[2] == "text").strip()) < 20 or is_mcq_page:
#                 try:
#                     pix = page.get_pixmap(dpi=600)
#                     img_filename = f"page_{j + 1}_full.png"
#                     img_path = os.path.join("static", img_filename)
#                     pix.save(img_path)
#                     img = Image.open(img_path).convert('L')
#                     img = ImageEnhance.Contrast(img).enhance(3.0)
#                     img = ImageEnhance.Sharpness(img).enhance(3.0)
#                     processed_path = os.path.join("static", f"processed_{img_filename}")
#                     img.save(processed_path)
#                     custom_config = r"--psm 6 -l eng"
#                     if is_mcq_page:
#                         custom_config = r"--psm 4 -c preserve_interword_spaces=1"
#                     ocr_data = pytesseract.image_to_data(
#                         Image.open(processed_path),
#                         config=custom_config,
#                         output_type=pytesseract.Output.DICT
#                     )
#                     for k in range(len(ocr_data['text'])):
#                         text = ocr_data['text'][k]
#                         if text.strip():
#                             x = ocr_data['left'][k]
#                             y = ocr_data['top'][k]
#                             blocks.append(((y, x), text, "ocr"))
#                     os.remove(processed_path)
#                 except Exception as e:
#                     print(f"[Error] OCR failed on page {j + 1}: {e}")
#
#             # Precise sorting using coordinates
#             try:
#                 blocks.sort(key=lambda x: (x[0][0], x[0][1]))
#                 page_text = []
#                 current_line_y = -1
#                 current_line = []
#                 for pos, text, block_type in blocks:
#                     y = pos[0]
#                     if current_line_y < 0 or abs(y - current_line_y) < 15:
#                         current_line.append(text)
#                         current_line_y = y
#                     else:
#                         page_text.append(" ".join(current_line))
#                         current_line = [text]
#                         current_line_y = y
#                 if current_line:
#                     page_text.append(" ".join(current_line))
#                 page_text = "\n".join(page_text)
#                 if is_mcq_page:
#                     page_text = re.sub(r'(\b[A-D]) (?=\w)', r'\1', page_text)
#                     page_text = re.sub(r'([A-D])\.\s*\n\s*', r'\1. ', page_text)
#             except Exception as e:
#                 print(f"Sorting failed: {e}")
#                 page_text = "\n".join(t[1] for t in blocks)
#
#             if page_text.strip():
#                 chunk_texts.append(f"=== PAGE {j + 1} ===\n{page_text}")
#         if chunk_texts:
#             full_text.append("\n\n".join(chunk_texts))
#         i += pages_per_chunk - overlap
#     doc.close()
#
    # # Save the output to a file
    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write("\n\n".join(full_text))
    # print(f"Successfully saved raw text to '{output_path}'.")


def format_mcq(mcq):
    return {
        "question": mcq.get("question") or mcq.get("ques") or mcq.get("q"),
        "noise" : mcq.get("noise"),
        "image": mcq.get("image") or mcq.get("img"),
        "options": mcq.get("options") or mcq.get("opts"),
        "answer": mcq.get("answer") or mcq.get("ans") or mcq.get("correct")
    }

# ===================================================
# uncomment the text below to use gemini pipeline instead of the pre-trained model
# ===================================================

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


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






UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==============================
#
# API
#
# ===============================

@app.route("/create_question_bank", methods=["POST"])
def upload_pdf():
    print("hello")
    user_id = request.form.get("userId")
    title = request.form.get("title")
    description = request.form.get("description")
    pdf_file = request.files.get("pdf")

    if not all([user_id, title, description, pdf_file]):
        return jsonify({"error": "userId, title, description, and pdf file are required"}), 400

    # Save PDF with original name + unique prefix
    original_name = secure_filename(pdf_file.filename)

    file_name = f"{original_name}"
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    pdf_file.save(file_path)

    createdAtTimestamp = datetime.now().isoformat()

    # Extract and format MCQs

#    final-data = extract_from_pdf(file_path) #uncomment this line and the rest of the code block when you want to use the gemini

#    final_data = process_pdf_to_json(file_path, labels_json_path="reference.json", model_dir="./CRF_BERT_MODEL")  ### UNCOMMENT THIS IF YOU WANT TO USE THE BERT_CRF MODEL

#    final_data = process_pdf_pipeline(file_path, model_path="./output_data/model.pt", vocab_path= "./output_data/vocabs.pkl") #UNCOMMENT THIS IF YOU WANT TO HAVE THE 'model.pt' and 'vocab.pkl' file with you and want to run the inference locally

    #raw_json_data, final_data, annotated_path = process_pdf_pipeline(input_pdf_path=file_path, model_path="./checkpoints/layoutlmv3_crf_new.pth") #UNCOMMENT THIS IF YOU WANT TO USE THE LAYOUTLMV3 script for inference

    final_data =  process_pdf_pipeline(file_path) #huggingface API

    indexed_mcqs = []
    for i, mcq in enumerate(final_data):
        mcq['documentIndex'] = i  # Add the index here
        indexed_mcqs.append(mcq)
    stored_id = store_mcqs(user_id, title, description, indexed_mcqs, file_name, createdAtTimestamp)


    return Response(
        json.dumps({
            "generatedQAId": stored_id,
            "userId":user_id,
            "fileName": file_name,
            "createdAt": createdAtTimestamp,

        }, ensure_ascii=False),
        mimetype="application/json"
    )




@app.route("/question_bank_by_user", methods=["POST"])
def paper_sets_by_userID():
    data = request.get_json(silent=True) or request.form.to_dict()
    userId = data.get("userId")

    mcqs_data = fetch_mcqs(userId=userId)
    if not mcqs_data:
        return jsonify({"message": "No Paper Sets found"})

    # FIX: Iterate through each paper set and sort its MCQs list
    for paper_set in mcqs_data:
        # Check if the 'mcqs' list exists and is iterable
        if paper_set.get('metadata', {}).get('mcqs'):
            mcqs_list = paper_set['metadata']['mcqs']


            # This handles older data that might have missing or None 'documentIndex' values.
            paper_set['metadata']['mcqs'] = sorted(
                mcqs_list,
                key=lambda x: int(x['documentIndex'])
                if x.get('documentIndex') is not None else float('inf')
            )
            # ===============================================

    return Response(
        json.dumps(mcqs_data, ensure_ascii=False, indent=4),
        mimetype="application/json"
    )


@app.route("/question_bank_by_id", methods=["POST"])
def paper_sets_by_generatedQAId():
   data = request.get_json(silent=True) or request.form.to_dict()
   generatedQAId = data.get("generatedQAId")

   if not generatedQAId:
    return jsonify({"error": "generatedQAId is required"}), 400

   results = fetch_mcqs(generatedQAId=generatedQAId)

   if not results:
       return jsonify({"error": "No MCQs found for the provided ID"}), 200

   if results and results[0].get('metadata', {}).get('mcqs'):
       mcqs_list = results[0]['metadata']['mcqs']
       # Sort by the 'documentIndex' field.
       # Fall back to 0 if the index is missing, though it shouldn't be.
       results[0]['metadata']['mcqs'] = sorted(
           mcqs_list,
           key=lambda x: x.get('documentIndex', 0)
       )
   # ===============================================
   # Return the full list of results, as generated by fetch_mcqs
   return jsonify(results)



@app.route("/generate_test", methods=["POST"])
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



@app.route("/combined_paperset", methods=["POST"])
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


@app.route("/paper_set/<testId>", methods=["GET"])
def testId(testId):
    """

    API to fetch a specific test session by its ID.
    """
    test_data = fetch_test_by_testId(testId)
    if not test_data:
        return jsonify({"error": "Test  not found"}), 200
    return jsonify(test_data), 200


@app.route("/paper_sets_by_user/<userId>", methods=["GET"])
def test_history_by_userId(userId):
    """
    API to fetch a list of test sessions for a given user, including all details.
    """
    test_history = test_sessions_by_userId(userId)
    if test_history is None:
        return jsonify({"error": "An error occurred while fetching test history"}), 500

    if not test_history:
        return jsonify({"message": "No test sessions found for this user"}), 200

    return jsonify(test_history), 200


@app.route("/submit_test", methods=["POST"])
def submit_test():
    """
    API to submit student answers and get results,
    and also to store the submission for future analysis.
    """
    payload = request.get_json(silent=True) or {}
    answers = payload.get("answers")
    userId = payload.get("userId")
    testId = payload.get("testId")
    testTitle = payload.get("testTitle")
    timeSpent = payload.get("timeSpent")
    totalTime = payload.get("totalTime")
    submittedAt = datetime.now().isoformat()
    totalQuestions = payload.get("totalQuestions")

    if not all([userId, testId, answers, submittedAt]):
        return jsonify({"error": "Missing required fields: userId, testId, answers"}), 400

    if not isinstance(answers, list):
        return jsonify({"error": "Answers must be a list"}), 400

    score = 0

    detailed_results = []

    for item in answers:
        question = item.get("question")
        submitted_answer = item.get("your_answer")
        correct_answer = item.get("correct_answer")

        if question and submitted_answer and correct_answer:
            is_correct = (submitted_answer == correct_answer)
            if is_correct:
                score += 1

            detailed_results.append(OrderedDict([
                ("question", question),
                ("your_answer", submitted_answer),
                ("correct_answer", correct_answer),
                ("is_correct", is_correct)
            ]))
        else:
            detailed_results.append(OrderedDict([
                ("question", question or "N/A"),
                ("error", "Missing required fields")
            ]))

    # Now, we store the results in the database
    is_stored = store_submitted_test(
        userId=userId,
        testId=testId,
        testTitle=testTitle,
        timeSpent=timeSpent,
        totalTime=totalTime,
        submittedAt=submittedAt,
        detailed_results=detailed_results,
        score=score,
        total_questions=totalQuestions
    )

    if not is_stored:
        return jsonify({"error": "Failed to store submission"}), 500

    response = OrderedDict([
        ("total_questions", totalQuestions),
        ("score", score),
        ("testTitle", testTitle),
        ("submittedAt", submittedAt),
        ("userId", userId),
        ("testId", testId),
        ("timeSpent", timeSpent),
        ("detailed_results", detailed_results)
    ])

    return jsonify(response)

@app.route("/submitted_tests/<userId>", methods=["GET"])
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



@app.route("/question_bank/<generatedQAId>/edit", methods=["PUT"])
def edit_question_bank(generatedQAId):
    """
    Unified API to perform add, edit, or delete operations on questions,
    and update the question bank's Title and Description.
    """
    payload = request.get_json(silent=True) or {}
    edits = payload.get("edits")

    # Extract metadata fields directly from the payload
    new_title = payload.get("title")
    new_description = payload.get("description")

    metadata_update_status = {"title_updated": False, "description_updated": False}

    # --- Step 1: Update Question Bank Metadata (Title/Description) ---
    if new_title is not None or new_description is not None:
        # Call the helper function to update only the metadata fields in the main collection
        metadata_update_status = update_question_bank_metadata(
            generatedQAId=generatedQAId,
            title=new_title,
            description=new_description
        )
        if not metadata_update_status.get("success", True):
            # If the update failed, return an error immediately
            return jsonify({"error": f"Failed to update metadata for Question Bank ID: {generatedQAId}"}), 500

    # --- Step 2: Process Question-level Edits ---
    if edits and isinstance(edits, list):
        for edit in edits:
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

    # --- Step 3: Fetch Updated Data and Respond ---
    updated_data = fetch_mcqs(generatedQAId=generatedQAId)

    if not updated_data:
        return jsonify({
            "error": "Update operations processed, but the question bank was not found.",
            "generatedQAId_used": generatedQAId
        }), 404

    updated_questions_count = len(updated_data[0].get("metadata", {}).get("mcqs", []))

    return jsonify({
        "message": "Question bank updated successfully",
        "title_updated": metadata_update_status["title_updated"],
        "description_updated": metadata_update_status["description_updated"],
        "updated_questions_count": updated_questions_count
    }), 200




@app.route("/create_manual_question_bank", methods=["POST"])
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
        mcq['documentIndex'] = i  # Assign sequential index
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



@app.route("/question_bank/<generatedQAId>", methods=["DELETE"])
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
        }), 404






@app.route("/submitted_test/<testId>", methods=["DELETE"])
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




@app.route("/paper_sets/<testId>", methods=["DELETE"])
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
            "message": f"Test session '{testId}' deleted successfully."
        }), 200


@app.route("/paper_sets/<testId>/edit", methods=["PUT"])
def edit_paperset(testId):
    """
    Unified API to edit an existing test session's metadata (e.g., totalTime, title)
    or its list of questions.

    Payload should contain the specific fields to update (e.g., {"testTitle": "New Title"}).
    If updating questions, the payload must contain the full, new 'questions' list.
    """
    payload = request.get_json(silent=True) or {}

    if not testId:
        return jsonify({"error": "testId is required"}), 400

    # 1. Fetch the existing test session record
    existing_record = fetch_test_by_testId(testId)
    if not existing_record:
        return jsonify({"error": f"Test session with ID '{testId}' not found."}), 404

    # 2. Merge existing data with new payload
    # Note: We must update the 'questions' list if provided in the payload
    updated_data = existing_record.copy()

    # Remove the existing 'testId' from metadata to avoid redundancy if it was there
    if 'testId' in updated_data:
        del updated_data['testId']

        # Merge the payload into the existing record
    for key, value in payload.items():
        if key in ['testTitle', 'totalTime', 'questions']:  # Keys we allow to be updated
            updated_data[key] = value

    # 3. Call the helper function to perform the ChromaDB update
    # The helper function must handle re-stringifying the 'questions' list if needed.
    success = update_test_session(testId, updated_data)

    if success:
        return jsonify({
            "message": "Test session updated successfully",
            "testId": testId,
            "updated_data_keys": list(payload.keys())
        }), 200
    else:
        return jsonify({"error": "Failed to update test session in database."}), 500




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)






