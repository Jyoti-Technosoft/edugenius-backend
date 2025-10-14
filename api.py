import uuid

#from examples.transformation.signature_method import model
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

from matplotlib.cbook import CallbackRegistry
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.Model import Model

pytesseract.pytesseract.tesseract_cmd = r"./usr/bin/tesseract"

app = Flask(__name__)
CORS(app)
from collections import OrderedDict
from vector_db import store_mcqs, fetch_mcqs,fetch_random_mcqs, store_test_session, fetch_test_session_by_testId,test_sessions_by_userId
from werkzeug.utils import secure_filename

#
# ==================================================
#
# GEMINI API FUNCTION

# uncomment this entire text block and use your API-KEY to use Gemini AI to extract question and answers
#this file is using our pre-trained model by default
#
# ===================================================
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




#
# import json
#
# def extract_from_pdf(pdf_path):
#     # This call now saves the file and does not return a list of chunks
#     extract_text_chunks(pdf_path, output_path="raw_text.txt")
#
#     # Read the content from the newly created file
#     try:
#         with open("raw_text.txt", 'r', encoding='utf-8') as f:
#             raw_text = f.read()
#     except FileNotFoundError:
#         print("[ERROR] raw_text.txt not found. Exiting.")
#         return []
#
#     # Process the raw text into chunks
#     chunks = raw_text.split("\n\n")
#
#     all_mcqs = []
#     seen = set()
#
#     for idx, chunk in enumerate(chunks):
#         if not chunk.strip():
#             continue
#
#         prompt = get_prompt(chunk)
#         try:
#             resp = model.generate_content(prompt)
#             raw = resp.text.strip()
#             print(f"[Gemini chunk {idx+1}]:", raw)
#
#             if raw.startswith("```json"):
#                 raw = raw.replace("```json", "").replace("```", "").strip()
#
#             try:
#                 data = json.loads(raw)
#                 mcqs_from_chunk = []
#
#                 if isinstance(data, list):
#                     # Handle the case where the response is a list of questions
#                     mcqs_from_chunk = data
#                 elif isinstance(data, dict) and "mcqs" in data:
#                     # Handle the case where the response is a dictionary with an 'mcqs' key
#                     mcqs_from_chunk = data["mcqs"]
#                 else:
#                     print(f"[Warning] Unexpected JSON format for chunk {idx+1}. Skipping.")
#                     continue
#
#                 for mcq in mcqs_from_chunk:
#                     question_text = mcq.get('question', '').strip()
#                     if question_text and question_text not in seen:
#                         all_mcqs.append(mcq)
#                         seen.add(question_text)
#
#             except json.JSONDecodeError as e:
#                 print(f"[Warning] Failed to parse JSON for chunk {idx+1}: {e}")
#                 print("Raw response:", raw)
#                 continue
#         except Exception as e:
#             print(f"[Error] Gemini content generation failed for chunk {idx+1}: {e}")
#             continue
#
#     return all_mcqs
#

# ===================================================
#
# PRE-TRAINED MODEL ( Comment this whole block out if you dont want to use it)
#
# ===================================================


import torch
import json
import re
import fitz
import os
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import torch.nn as nn
from TorchCRF import CRF
from typing import List, Dict, Any
from transformers import DistilBertTokenizerFast, DistilBertModel


# Set the Tesseract executable path if it's not in your PATH
# On Windows, it might be something like: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


# --- Custom Model Class (Must be included to load the saved state) ---
class DistilBertCrfForTokenClassification(nn.Module):
    def __init__(self, num_labels, model_name="distilbert-base-uncased"):
        super(DistilBertCrfForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.distilbert.config.seq_classif_dropout)
        self.classifier = nn.Linear(self.distilbert.config.dim, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            crf_mask = labels != -100
            labels[labels == -100] = 0
            crf_loss = -self.crf(logits, labels, crf_mask).sum()
            return {"loss": crf_loss, "logits": logits}

        return {"logits": logits}

    def predict(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        viterbi_path = self.crf.viterbi_decode(logits, attention_mask.bool())
        return viterbi_path


# --- Main Pipeline Function ---
def process_pdf_to_json(pdf_path, labels_json_path, model_dir):
    """
    Automates the entire workflow:
    1. Extracts and transforms text from a PDF with header/footer removal and OCR fallback.
    2. Runs predictions on the transformed text.
    3. Converts the predictions to structured JSON and returns it.
    """
    try:
        doc = fitz.open(pdf_path)
    except FileNotFoundError:
        print(f"Error: The PDF file '{pdf_path}' was not found.")
        return None

    full_text_parts = []
    os.makedirs("static", exist_ok=True)

    for j in range(len(doc)):
        page = doc[j]
        blocks = []
        is_mcq_page = False

        header_height = 50
        footer_height = 50
        content_rect = fitz.Rect(0, header_height, page.rect.width, page.rect.height - footer_height)

        try:
            page_dict = page.get_text("dict", sort=True, clip=content_rect)
            for block in page_dict.get("blocks", []):
                bbox = block.get("bbox", [0, 0, 0, 0])
                sort_key = (bbox[1], bbox[0])
                if block.get("type") == 0:
                    text_block = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_block += span.get("text", "")
                        text_block += "\n"
                    text_block = text_block.strip()
                    if text_block:
                        if re.search(r'^[A-D][\.\)]\s', text_block, re.MULTILINE):
                            is_mcq_page = True
                        blocks.append((sort_key, text_block, "text"))
        except Exception as e:
            print(f"[Warning] Failed to parse layout on page {j + 1}: {e}")

        page_text = ""
        if len(" ".join(t[1] for t in blocks if t[2] == "text").strip()) < 20 or is_mcq_page:
            try:
                print(f"Running OCR on page {j + 1}...")
                pix = page.get_pixmap(dpi=300)
                img_path = f"static/temp_page_{j + 1}.png"
                pix.save(img_path)
                image = Image.open(img_path)

                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2)

                cropped_image = image.crop((content_rect.x0, content_rect.y0, content_rect.x1, content_rect.y1))

                page_text = pytesseract.image_to_string(cropped_image)
                os.remove(img_path)
            except Exception as e:
                print(f"[Error] OCR failed on page {j + 1}: {e}")
                page_text = ""
        else:
            vertical_gaps = []
            if len(blocks) > 1:
                sorted_blocks = sorted([b for b in blocks if b[2] == "text"], key=lambda x: (x[0][0], x[0][1]))
                if len(sorted_blocks) > 1:
                    for i in range(1, len(sorted_blocks)):
                        gap = sorted_blocks[i][0][0] - sorted_blocks[i - 1][0][0]
                        if gap > 0: vertical_gaps.append(gap)

            if vertical_gaps:
                average_gap = np.mean(vertical_gaps)
                LINE_BREAK_THRESHOLD = average_gap * 1.5
                PARAGRAPH_BREAK_THRESHOLD = average_gap * 2.5
            else:
                LINE_BREAK_THRESHOLD = 15
                PARAGRAPH_BREAK_THRESHOLD = 30

            try:
                blocks.sort(key=lambda x: (x[0][0], x[0][1]))
                page_text_parts = []
                current_line_y = -1
                for pos, text, block_type in blocks:
                    y = pos[0]
                    if current_line_y > 0:
                        vertical_gap = abs(y - current_line_y)
                        if vertical_gap > PARAGRAPH_BREAK_THRESHOLD:
                            page_text_parts.append("[PARAGRAPH_BREAK]")
                        elif vertical_gap > LINE_BREAK_THRESHOLD:
                            page_text_parts.append("[BR]")
                    page_text_parts.append(text)
                    current_line_y = y
                page_text = " ".join(page_text_parts)
                if is_mcq_page:
                    page_text = re.sub(r'(\b[A-D]) (?=\w)', r'\1', page_text)
                    page_text = re.sub(r'([A-D])\.\s*\n\s*', r'\1. ', page_text)
            except Exception as e:
                print(f"Sorting failed: {e}")
                page_text = " ".join(t[1] for t in blocks)

        if page_text.strip():
            full_text_parts.append(f"=== PAGE {j + 1} ===\n{page_text}")

    doc.close()

    raw_text = "\n\n".join(full_text_parts)

    # --- MODIFIED: More flexible regex to match various question formats ---
    question_blocks = re.split(r'((?:Q\.\d+|Question\s+\d+|\d+\.)(?:\s|$))', raw_text, flags=re.IGNORECASE)
    # --- END MODIFIED SECTION ---

    transformed_blocks = []
    if len(question_blocks) > 1:
        # The first element is always the text before the first match, so we start from index 1.
        # This modification handles cases where the first question isn't perfectly at the start of the text.
        for i in range(1, len(question_blocks), 2):
            question_num = question_blocks[i].strip()
            rest_of_block = question_blocks[i + 1].strip()

            clean_block = re.sub(r'\[BR\]|\[PARAGRAPH_BREAK\]|=== PAGE \d+ ===', '', rest_of_block)
            clean_block = re.sub(r'\s+', ' ', clean_block).strip()
            clean_block = clean_block.replace('(A)', 'a.').replace('(B)', 'b.').replace('(C)', 'c.').replace('(D)',
                                                                                                             'd.')
            clean_block = clean_block.replace('(a)', 'a.').replace('(b)', 'b.').replace('(c)', 'c.').replace('(d)',
                                                                                                             'd.')
            clean_block = clean_block.replace(' a)', ' a.').replace(' b)', ' b.').replace(' c)', ' c.').replace(' d)',
                                                                                                             ' d.')
            clean_block = clean_block.replace('(i)', 'a.').replace('(ii)', 'b.').replace('(iii)', 'c.').replace('(iv)',
                                                                                                                'd.')
            clean_block = clean_block.replace(' i ', 'a.').replace(' ii ', 'b.').replace(' iii ', 'c.').replace(' iv ',
                                                                                                                'd.')
            clean_block = clean_block.replace('Ans:', '')
            clean_block = clean_block.replace('ANSWER:','')
            clean_block = clean_block.replace('Answer:', '')

            transformed_blocks.append(f"{question_num} {clean_block}")

    final_mcqs_text = "\n".join(transformed_blocks)

    try:
        with open(labels_json_path, 'r', encoding='utf-8') as f:
            uploaded_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Labels file '{labels_json_path}' not found.")
        return None

    unique_labels = sorted(list(set(item['label'] for item in uploaded_data)))
    id_to_label = {i: label for i, label in enumerate(unique_labels)}

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertCrfForTokenClassification(num_labels=len(unique_labels))
    model.load_state_dict(torch.load(f"{model_dir}/model.pt"))
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    lines = final_mcqs_text.strip().split('\n')
    predictions_output = []

    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line: continue
        tokens = re.findall(r"[\w']+|[.,?!:;()]", line)
        tokenized_input = tokenizer(tokens, return_tensors="pt", truncation=True, is_split_into_words=True)
        input_ids = tokenized_input["input_ids"].to(device)
        attention_mask = tokenized_input["attention_mask"].to(device)

        with torch.no_grad():
            predictions = model.predict(input_ids=input_ids, attention_mask=attention_mask)

        word_ids = tokenized_input.word_ids(batch_index=0)
        predicted_list = []
        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                predicted_list.append({"token": tokens[word_idx], "label": id_to_label[predictions[0][idx]]})
            previous_word_idx = word_idx

        predictions_output.append({"line_number": line_number, "text": line, "predictions": predicted_list})

    structured_data = []
    for item in predictions_output:
        predictions = item.get("predictions", [])
        state = "question"
        current_question, current_options, current_option_text, current_option_key, current_answer = [], {}, [], None, []

        for pred in predictions:
            token, label = pred["token"], pred["label"]
            if "QUESTION" in label:
                state = "question"
                current_question.append(token)
            elif "OPTION" in label:
                if label.startswith("B-"):
                    if current_option_key is not None: current_options[current_option_key] = " ".join(
                        current_option_text).strip()
                    state = "option"
                    current_option_key = token.upper()
                    current_option_text = []
                elif label.startswith("I-") and state == "question":
                    state = "option"
                    current_option_key = "A"
                    current_option_text.append(token)
                else:
                    current_option_text.append(token)
            elif "ANSWER" in label:
                if current_option_key is not None:
                    current_options[current_option_key] = " ".join(current_option_text).strip()
                    current_option_text = []
                    current_option_key = None
                state = "answer"
                current_answer.append(token)

        if current_option_key is not None: current_options[current_option_key] = " ".join(current_option_text).strip()
        for key, value in current_options.items():
            if value.startswith(f"{key.lower()}."): current_options[key] = value[2:].strip()

        if current_question:
            structured_data.append({
                "question": " ".join(current_question).strip(),
                "options": current_options,
                "answer": " ".join(current_answer).strip()
            })

    return structured_data

# ===========================================================
#
# PRE_TRAINED MODEL PIPELINE END
#
# ===========================================================


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
#    mcqs = extract_from_pdf(file_path) #activate this line and the rest of the code block when you want to use the gemini
    mcqs = process_pdf_to_json(file_path, labels_json_path="reference.json", model_dir="./CRF_BERT_MODEL")
    #stored_count = store_mcqs(user_id, title, description, mcqs, file_name)
    stored_id = store_mcqs(user_id, title, description, mcqs, file_name, createdAtTimestamp)




    return Response(
        json.dumps({
            "generatedQAId": stored_id,
            "userId":user_id,
            "fileName": file_name,
            "createdAt": createdAtTimestamp,

        }, ensure_ascii=False),
        mimetype="application/json"
    )






@app.route("/paper_sets_by_user", methods=["POST"])
def paper_sets_by_userID():
    data = request.get_json(silent=True) or request.form.to_dict()
    userId = data.get("userId")

    mcqs_data = fetch_mcqs(userId=userId)
    if not mcqs_data:
        return jsonify({"message": "No Paper Sets found"})

    return Response(
        json.dumps(mcqs_data, ensure_ascii=False, indent=4),
        mimetype="application/json"
     )



@app.route("/paper_sets_by_id", methods=["POST"])
def paper_sets_by_generatedQAId():
   data = request.get_json(silent=True) or request.form.to_dict()
   generatedQAId = data.get("generatedQAId")

   if not generatedQAId:
    return jsonify({"error": "generatedQAId is required"}), 400

   results = fetch_mcqs(generatedQAId=generatedQAId)

   if not results:
       return jsonify({"error": "No MCQs found for the provided ID"}), 404

   # Return the full list of results, as generated by fetch_mcqs
   return jsonify(results)


@app.route("/generate_test", methods=["POST"])
def generate_test():
    """
    API to fetch MCQs by generated-qa-Id and marks (limit),
    and also to create a new test entry.

    {
  "generatedQAId": "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f6a1b2",
  "marks": 10,
  "userId": "user12345",
  "testTitle": "My First Test"
}

It will create a test for the specified number of marks and create a corresponding testId

    """
    data = request.get_json(silent=True) or request.form

    generatedQAId = data.get("generatedQAId")
    marks = data.get("marks")
    userId = data.get("userId")
    testTitle = data.get("testTitle")
    testTimeLimit = data.get("testTimeLimit")

    if not generatedQAId:
        return jsonify({"error": "generatedQAId is required"}), 400

    if not marks:
        return jsonify({"error": "marks is required"}), 400

    # testTitle is required to store the test
    if not testTitle:
        return jsonify({"error": "testTitle is required"}), 400

    try:
        marks = int(marks)
    except ValueError:
        return jsonify({"error": "marks must be an integer"}), 400

    # Generate a unique test ID and a timestamp
    testId = str(uuid.uuid4())
    createdAt = datetime.now().isoformat()

    # Use the fetch_random_mcqs function from the provided vector_db module.
    # It returns a list containing a single dictionary with 'metadata' and 'mcqs'
    test_data_results = fetch_random_mcqs(generatedQAId, num_questions=marks)

    if not test_data_results:
        return jsonify({"message": "No MCQs found"}), 404

    # Extract the MCQs list from the results
    mcqs_data = test_data_results[0].get("metadata", {}).get("mcqs", [])

    # The functionality to store the test with testId and createdAt
    # is not available in the provided vector_db.py file.
    # A new function, for example, `vector_db.store_test_session(...)`,
    # would be needed to save this information.

    # Fix: Pass the correct arguments in the correct order
    if userId:
        is_stored = store_test_session(userId, testId, testTitle,testTimeLimit, createdAt, mcqs_data)
        if not is_stored:
            return jsonify({"error": "Failed to store test session"}), 500

    # Return the test ID along with the MCQs
    return jsonify({
        "message": "Test created and stored successfully",
        "userId": userId,
        "testId": testId,
        "testTimeLimit": testTimeLimit,
        "createdAt": createdAt,
        "questions": mcqs_data
    }), 200

@app.route("/combined_test", methods=["POST"])
def combined_test():

    data = request.get_json(silent=True) or request.form

    userId = data.get("userId")
    testTitle = data.get("testTitle")
    testTimeLimit = data.get("testTimeLimit")
    total_questions = data.get("total_questions")
    sources = data.get("sources")

    # Validate required inputs
    if not all([userId, testTitle, testTimeLimit, total_questions, sources]) or not isinstance(sources, list):
        return jsonify({"error": "userId, testTitle, total_questions, and a list of sources are required"}), 400

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

        # Fetch a random sample from this source, filtered by userId
        mcqs_record = fetch_random_mcqs(generatedQAId=qa_id, num_questions=num_questions)
        if mcqs_record:
            all_mcqs.extend(mcqs_record[0].get("metadata", {}).get("mcqs", []))

    # Shuffle the combined list of all MCQs
    random.shuffle(all_mcqs)

    if not all_mcqs:
        return jsonify({"message": "No MCQs found for the provided IDs"}), 404

    # Generate test metadata
    testId = str(uuid.uuid4())
    createdAt = datetime.now().isoformat()

    # Store the test session with the new title
    store_test_session(userId, testId, testTitle, testTimeLimit, createdAt, all_mcqs)

    return jsonify({

        "userId": userId,
        "testId": testId,
        "testTitle": testTitle,
        "testTimeLimit": testTimeLimit,
        "createdAt": createdAt,
        "questions": all_mcqs
    }), 200


@app.route("/test/<testId>", methods=["GET"])
def testId(testId):
    """


    API to fetch a specific test session by its ID.
    """
    test_data = fetch_test_session_by_testId(testId)
    if not test_data:
        return jsonify({"error": "Test session not found"}), 404
    return jsonify(test_data), 200


@app.route("/test_history/<userId>", methods=["GET"])
def test_history_by_userId(userId):
    """
    API to fetch a list of test sessions for a given user, including all details.
    """
    test_history = test_sessions_by_userId(userId)
    if test_history is None:
        return jsonify({"error": "An error occurred while fetching test history"}), 500

    if not test_history:
        return jsonify({"message": "No test sessions found for this user"}), 404

    return jsonify(test_history), 200






@app.route("/submit_test", methods=["POST"])
def submit_test():
    """
    API to submit student answers and get results.

    The request body is a list of objects, each containing a question,
    the submitted answer, and the correct answer for verification.

    {
  "userId": "user12345",
  "testId": "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f6a1b2",
  "testTitle": "History Quiz 1",
  "answers": [
    {
      "question": "What is the capital of France?",
      "your_answer": "Paris",
      "correct_answer": "Paris"
    },
    {
      "question": "What is 2 + 2?",
      "your_answer": "5",
      "correct_answer": "4"
    },
    {
      "question": "Which planet is known as the Red Planet?",
      "your_answer": "Mars",
      "correct_answer": "Mars"
    }
  ]
}
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "answers" not in payload:
        return jsonify({"error": "Request body must be a dictionary with an 'answers' key"}), 400

    answers = payload.get("answers")
    userId = payload.get("userId")
    testTitle = payload.get("testTitle")
    testId = payload.get("testId")
    submittedAt = datetime.now().isoformat()
    testTime = payload.get("testTime")




    if not isinstance(answers, list):
        return jsonify({"error": "Request body must be a list of answers"}), 400

    score = 0
    total_questions = len(answers)
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
            # Handle malformed or missing data in an entry
            detailed_results.append(OrderedDict([
                ("question", question or "N/A"),
                ("error", "Missing required fields")
            ]))

    response = OrderedDict([
        ("total_questions", total_questions),
        ("score", score),
        ("testTitle", testTitle),
        ("submittedAt", submittedAt),
        ("userId", userId),
        ("testId", testId),
        ("testTime", testTime),
        ("detailed_results", detailed_results)
    ])

    return jsonify(response)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)








