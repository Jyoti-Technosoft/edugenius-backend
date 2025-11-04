#Not Required File
import fitz
import os
import re
import json
import tempfile
import shutil
from PIL import Image, ImageEnhance
import pytesseract
from fpdf import FPDF as FPDF_BASE  # Use FPDF_BASE to support the new PDF class logic
import torch
import torch.nn as nn
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Model
from TorchCRF import CRF
import time
import base64

# Define the directory for saving images (Used by extract_text_chunks)
IMAGE_OUTPUT_DIR = "extracted_images"


# ==============================================================================
# FPDF Helper Class for Unicode Support
# ==============================================================================

class PDF(FPDF_BASE):
    """Custom FPDF class with fallback font setup."""

    def __init__(self):
        super().__init__()
        # We try to use a Unicode font, but if the file is missing, we use Arial.
        try:
            # uni=True is the key for Unicode support in fpdf
            self.add_font('DejaVuSans', '', 'DejaVuSansCondensed.ttf', uni=True)
            self.set_font('DejaVuSans', '', 12)
        except Exception:
            # Fallback to default Arial if custom font file is not found
            self.set_font('Arial', '', 12)

        # --- CRITICAL FIX: Set core document encoding to UTF-8 (Required by the original script) ---
        self.set_doc_option('core_fonts_encoding', 'utf-8')
        # ---------------------------------------------------------


# ==============================================================================
# PART 1: PDF Text Extraction Function (UPDATED)
# ==============================================================================

def extract_text_chunks(pdf_path):
    """
    PDF extractor that reads the PDF, cleans the text, filters out headers/page
    numbers based on position, extracts and tags images using robust block access,
    and returns the single, combined text content.
    """
    start_time = time.time()

    # Ensure the image output directory exists
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    LINE_BREAK_THRESHOLD = 15
    PARAGRAPH_BREAK_THRESHOLD = 30

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return "", 0

    num_pages = len(doc)
    final_combined_text = []
    global_image_counter = 1

    HEADER_Y_MAX = 50
    FOOTER_Y_MIN = 700

    for page_num in range(num_pages):
        page = doc[page_num]
        blocks = []
        is_mcq_page = False

        # --- Content Extraction (PyMuPDF) ---
        try:
            page_dict = page.get_text("dict", sort=True)
            for block in page_dict.get("blocks", []):
                bbox = block.get("bbox", [0, 0, 0, 0])
                sort_key = (bbox[1], bbox[0])

                if block.get("type") == 0:  # Text block
                    text_block = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_block += span.get("text", "")
                        text_block += "\n"
                    text_block = text_block.strip()

                    if text_block:
                        # 1. NEW FILTERING LOGIC: Skip known noise (Headers/Page Numbers)
                        if (bbox[1] < HEADER_Y_MAX or bbox[3] > FOOTER_Y_MIN):
                            if re.fullmatch(r'(\d+|[ivx]+|\s*\bPage\s*\d+\b\s*|.{1,15})', text_block, re.IGNORECASE):
                                continue

                        # 2. Check for MCQ indicators
                        if re.search(r'^[A-D][\.\)]\s', text_block, re.MULTILINE):
                            is_mcq_page = True

                        blocks.append((sort_key, text_block, "text"))

                elif block.get("type") == 1:  # Image block: Extract and Tag
                    try:
                        # Attempt 1: Check for the 'image' key directly (works for simple inline images)
                        img_bytes = block.get("image")
                        img_ext = block.get("ext", "png")

                        # Attempt 2: Use doc.extract_image if xref is available (more reliable for complex images)
                        if not img_bytes:
                            xref = block.get("xref")
                            if xref:
                                img_data = doc.extract_image(xref)
                                img_bytes = img_data.get("image")
                                img_ext = img_data.get("ext", "png")

                        if img_bytes:
                            img_filename = f"image_{global_image_counter}.{img_ext}"
                            img_path = os.path.join(IMAGE_OUTPUT_DIR, img_filename)

                            # Save image to file
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)

                            # Insert the unique tag into the text stream
                            img_marker = f"[IMAGE: {img_filename}]"
                            blocks.append((sort_key, img_marker, "image"))
                            global_image_counter += 1
                        else:
                            # Print a warning if image type 1 block has no bytes or xref
                            print(
                                f"[Warning] Block Type 1 found but no image data/xref available on page {page_num + 1}")

                    except Exception as e:
                        # General exception catcher for unexpected failures
                        print(f"[Error] Critical failure saving image on page {page_num + 1}: {e}")

        except Exception:
            pass

        # --- OCR Fallback (Single Page) ---
        if len(" ".join(t[1] for t in blocks if t[2] != "image").strip()) < 20 or is_mcq_page:
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
                img = ImageEnhance.Contrast(img).enhance(3.0)
                img = ImageEnhance.Sharpness(img).enhance(3.0)

                custom_config = r"--psm 6 -l eng"
                if is_mcq_page:
                    custom_config = r"--psm 4 -c preserve_interword_spaces=1"

                ocr_data = pytesseract.image_to_data(
                    img,
                    config=custom_config,
                    output_type=pytesseract.Output.DICT
                )

                for k in range(len(ocr_data['text'])):
                    text = ocr_data['text'][k]
                    if text.strip():
                        x = ocr_data['left'][k]
                        y = ocr_data['top'][k]
                        blocks.append(((y, x), text, "ocr"))
            except Exception:
                pass

        # --- Precise Sorting and Custom Line Break Tokens ---
        try:
            # Sorts both text and image markers by position
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
        except Exception:
            page_text = " ".join(t[1] for t in blocks)

        if page_text.strip():
            final_combined_text.append(page_text)

    doc.close()

    final_text = "\n\n".join(final_combined_text)

    # Apply all global cleaning/regex to the single, full text block
    final_text = final_text.replace("[PARAGRAPH_BREAK]", "\n\n\n")
    final_text = final_text.replace("[BR]", "\n\n\n\n ")
    final_text = re.sub(r'([\.])(\w)', r'\1   \2', final_text)
    final_text = re.sub(r'([\)\]\}:])(\w)', r'\1   \2', final_text)

    # Replace line-separated numbers/roman numerals with the '\nQ ' marker
    final_text = re.sub(r'\n\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\s*\n', '\nQ.1 ', final_text)
    final_text = re.sub(r'\n\s*\d+\s*\n', '\nQ ', final_text)

    #  Replace Unicode Smart Quotes and other symbols ---
    final_text = final_text.replace('\u2018', "'").replace('\u2019', "'")
    final_text = final_text.replace('\u201c', '"').replace('\u201d', '"')
    final_text = final_text.replace('\u2026', '...')  # Ellipsis '…'
    final_text = final_text.replace('\u2013', '-')  # En dash '–'
    final_text = final_text.replace('\u2014', '--')  # Em dash '—'

    final_text = re.sub(r'\n\s*\n\s*\n', '\n\n', final_text).strip()
    end_time = time.time()
    duration = end_time - start_time

    # Return the extracted text and duration for the next script to use
    return final_text, duration


# ==============================================================================
# PART 2: PDF Regeneration Function (UPDATED)
# ==============================================================================

def create_pdf_from_text_with_markers(text_content, output_pdf_path):
    """
    Creates a new PDF with dynamic page breaks: attempts to fit 5 questions,
    but checks for potential overflow before writing Q4 and Q5 to ensure
    continuity (reverts to 4 or 3 questions per page if needed).
    """
    start_time = time.time()
    if not text_content:
        raise ValueError("Text content is empty. Cannot create PDF.")

    regex_delimiters = [
        r'(Q\.\s*\d+)',
        r'(\n|^)(\d+[\.\)])',
    ]

    try:
        # --- Preparation: Group content into question blocks and filter noise ---
        regex_pattern = '|'.join(regex_delimiters)
        sections = re.split(regex_pattern, text_content, flags=re.MULTILINE)

        question_blocks = []
        is_first_section = True

        for section in sections:
            if not section or not section.strip():
                continue

            stripped_section = section.strip()

            is_new_question_start = re.match(r'^(Q\.\s*\d+|\d+[\.\)])', stripped_section)
            is_potential_noise = len(stripped_section) < 20

            if is_new_question_start or is_first_section:
                # Start of a new question or the very first section (header)
                question_blocks.append(stripped_section)
                is_first_section = False
            elif is_potential_noise and question_blocks:
                # Merge short noise (answers, page numbers, etc.) into the previous block
                question_blocks[-1] += ' ' + stripped_section
            else:
                # Continuation of the previous question
                question_blocks[-1] += '\n' + stripped_section

        # Determine the index of the first actual question for header handling
        first_question_block_index = -1
        for idx, block in enumerate(question_blocks):
            if re.match(r'^(Q\.\s*\d+|\d+[\.\)])', block):
                first_question_block_index = idx
                break

        # --- Start Pagination ---
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)

        questions_on_current_page = 0
        current_block_index = 0

        # Y-Thresholds for forcing a break. These values are optimized for A4 paper (297 high).
        TIGHT_BREAK_Y = 200  # If Y is this high, Q4 will overflow (forcing break after Q3).
        LOOSE_BREAK_Y = 200  # If Y is this high, Q5 will overflow (forcing break after Q4).

        while current_block_index < len(question_blocks):

            block = question_blocks[current_block_index]
            is_new_question_block = re.match(r'^(Q\.\s*\d+|\d+[\.\)])', block)
            text_to_write = block

            # 1. Handle Header/Initial Block (if separated from Q1)
            if current_block_index < first_question_block_index:
                # Header content is written without counting
                pass

                # 2. Handle Question Blocks
            elif is_new_question_block:
                questions_on_current_page += 1
                text_to_write = '\n' + block  # Add newline before a new question

                # --- Conditional Page Break Logic (Check before writing the block) ---

                # A. CHECK POINT FOR Q4 (Counter hits 4)
                if questions_on_current_page == 4:
                    # If Q1-Q3 were huge (Y > TIGHT_BREAK_Y), force break after Q3.
                    if pdf.get_y() > TIGHT_BREAK_Y:
                        pdf.add_page()
                        # Q4 is now the first question on the new page
                        questions_on_current_page = 1

                        # B. CHECK POINT FOR Q5 (Counter hits 5)
                elif questions_on_current_page == 5:
                    # If Q1-Q4 were too long (Y > LOOSE_BREAK_Y), force break after Q4.
                    if pdf.get_y() > LOOSE_BREAK_Y:
                        pdf.add_page()
                        # Q5 is now the first question on the new page
                        questions_on_current_page = 1

                        # C. NORMAL BREAK (After Q5, when counter hits 6)
                elif questions_on_current_page > 5:
                    # Q1-Q5 fit safely. Break now before writing Q6.
                    pdf.add_page()
                    # Q6 is the first question on the new page
                    questions_on_current_page = 1

                    # 3. Write the Block Content
            if text_to_write.strip():
                # NOTE: Ensure you are using the 'text' parameter if running fpdf2
                pdf.multi_cell(0, 5, text=text_to_write)
                pdf.ln(2)

            current_block_index += 1  # Ensure the index always advances

        # Final output
        pdf.output(output_pdf_path, 'F')

        end_time = time.time()
        duration = end_time - start_time
        return duration

    except Exception as e:
        raise IOError(f"Failed to create output PDF at {output_pdf_path}: {e}")


# ==============================================================================
# PART 3: LayoutLMv3 + CRF Inference Model and Function
# ==============================================================================

class LayoutLMv3CRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        # Load pre-trained LayoutLMv3 (ensure transformers is installed)
        self.layoutlm = LayoutLMv3Model.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.layoutlm.config.hidden_size, num_labels)
        # TorchCRF must be installed
        self.crf = CRF(num_labels)

    def forward(self, input_ids=None, bbox=None, attention_mask=None, labels=None):
        outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=attention_mask.bool())
            return -log_likelihood.mean()
        else:
            best_paths = self.crf.viterbi_decode(emissions, mask=attention_mask.bool())
            return best_paths


def run_inference_on_pdf(pdf_path, model_path, temp_dir, annotated_pdf_path):
    labels = ["O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION", "B-ANSWER", "I-ANSWER"]
    id2label = {i: l for i, l in enumerate(labels)}
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3CRF("microsoft/layoutlmv3-base", num_labels=len(labels))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    # Load trained weights on CPU first
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # --- SAFE DYNAMIC QUANTIZATION ---
    # Only quantize Linear layers inside the classifier and CRF; skip LayoutLMv3 embeddings & special buffers
    model.classifier = torch.quantization.quantize_dynamic(
        model.classifier, {torch.nn.Linear}, dtype=torch.qint8
    )
    # NOTE: Do not quantize layoutlm or CRF layer

    # Move entire model to device
    model.to(device)

    try:
        src_doc = fitz.open(pdf_path)
    except Exception as e:
        raise IOError(f"Error opening regenerated PDF for inference: {e}")

    all_predictions = []
    annotated_doc = fitz.open()
    tesseract_config = '--psm 6'

    total_inference_time = 0
    page_inference_times = []

    for page_num in range(len(src_doc)):
        start_page_time = time.time()

        page = src_doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        tmp_img_path = os.path.join(temp_dir, f"tmp_page_{page_num + 1}.png")
        pix.save(tmp_img_path)
        pil_img = Image.open(tmp_img_path).convert("RGB")

        page_width = page.bound().width
        page_height = page.bound().height

        # --- OCR to get words & bboxes ---
        ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=tesseract_config)
        words, bboxes_raw = [], []
        for i in range(len(ocr_data["text"])):
            txt = (ocr_data["text"][i] or "").strip()
            if txt:
                words.append(txt)
                left, top, width, height = map(int, [ocr_data["left"][i], ocr_data["top"][i],
                                                     ocr_data["width"][i], ocr_data["height"][i]])
                bboxes_raw.append([left, top, left + width, top + height])

        if not words:
            new_page = annotated_doc.new_page(width=page_width, height=page_height)
            new_page.insert_image(new_page.rect, filename=tmp_img_path)
            os.remove(tmp_img_path)
            page_inference_times.append(time.time() - start_page_time)
            continue

        # --- Normalize bounding boxes ---
        x_scale = page_width / pix.width
        y_scale = page_height / pix.height
        normalized_bboxes = [
            [
                max(0, min(1000, int(1000 * ((b[0] * x_scale) / page_width)))),
                max(0, min(1000, int(1000 * ((b[1] * y_scale) / page_height)))),
                max(0, min(1000, int(1000 * ((b[2] * x_scale) / page_width)))),
                max(0, min(1000, int(1000 * ((b[3] * y_scale) / page_height))))
            ]
            for b in bboxes_raw
        ]

        # --- Encode & inference ---
        encoding = tokenizer(words, boxes=normalized_bboxes, return_tensors="pt", truncation=True, padding=True)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            preds = model(
                input_ids=encoding["input_ids"],
                bbox=encoding["bbox"],
                attention_mask=encoding["attention_mask"]
            )
            pred_token_ids = preds[0] if isinstance(preds, (list, tuple)) else preds

        token_to_word = tokenizer(words, boxes=normalized_bboxes, return_tensors="pt", truncation=True,
                                  padding=True).word_ids(batch_index=0)

        final_preds, previous_word_idx = [], None
        for token_idx, word_id in enumerate(token_to_word):
            if word_id is None:
                continue
            if word_id != previous_word_idx:
                try:
                    label_id = pred_token_ids[token_idx]
                    label_name = id2label.get(label_id, "O")
                except Exception:
                    label_name = "O"
                final_preds.append(label_name)
            previous_word_idx = word_id

        if len(final_preds) != len(words):
            final_preds = ["O"] * len(words)

        # --- Collect predictions ---
        page_results = [{"word": w, "bbox": b, "predicted_label": lab} for w, b, lab in
                        zip(words, bboxes_raw, final_preds)]
        all_predictions.extend(page_results)

        # --- Annotated PDF page ---
        new_page = annotated_doc.new_page(width=page_width, height=page_height)
        new_page.insert_image(new_page.rect, filename=tmp_img_path)
        for w, b, lab in zip(words, bboxes_raw, final_preds):
            if lab == "O":
                continue
            rect = fitz.Rect(b[0] * x_scale, b[1] * y_scale, b[2] * x_scale, b[3] * y_scale)
            new_page.draw_rect(rect, color=(1, 0, 0), width=1)
            new_page.insert_text((b[0] * x_scale, max(0, b[1] * y_scale - 8)), lab, fontsize=7, color=(0, 0, 1))

        os.remove(tmp_img_path)
        duration = time.time() - start_page_time
        page_inference_times.append(duration)
        total_inference_time += duration

    src_doc.close()
    annotated_doc.save(annotated_pdf_path)
    annotated_doc.close()

    return all_predictions, total_inference_time, page_inference_times


# ==============================================================================
# PART 4: Post-Processing and Filtering
# ==============================================================================


def convert_bio_to_structured_json(predictions: list):
    """
    Transforms a raw prediction list with BIO encoding into a cleaner,
    structured JSON format (as a Python list of dicts).
    """
    start_time = time.time()
    structured_data = []
    current_item = None
    current_option_key = None

    # --- FIX: Initialize noise_buffer here to ensure it's always defined ---
    noise_buffer = ""
    # ----------------------------------------------------------------------

    for item in predictions:
        word = item['word'].strip()
        if not word: continue
        label = item['predicted_label']

        # Skip tokens that are part of an entity but don't have a starting token.
        if label.startswith('I-') and current_item is None:
            continue

        if label.startswith('B-'):
            if current_item is not None and label == 'B-QUESTION':
                # Save the previous question before starting a new one
                structured_data.append(current_item)
                current_item = None

            entity_type = label[2:]
            if entity_type == 'QUESTION':
                current_item = {
                    'question': word,
                    'options': {},
                    'answer': '',
                    # Use the initialized or accumulated noise
                    'noise': noise_buffer.strip()
                }
                noise_buffer = ""  # Clear the buffer after assignment

                current_option_key = None
            elif entity_type == 'OPTION':
                if current_item is not None:
                    current_option_key = word
                    current_item['options'][current_option_key] = ''
            elif entity_type == 'ANSWER':
                if current_item is not None:
                    current_item['answer'] = word
                current_option_key = None

        elif label.startswith('I-'):
            # Clear noise buffer when a structured entity starts
            noise_buffer = ""
            entity_type = label[2:]
            if entity_type == 'QUESTION':
                if current_item is not None:
                    current_item['question'] += f' {word}'
            elif entity_type == 'OPTION':
                if current_option_key is not None and current_item is not None:
                    # Append to the current option's value
                    current_item['options'][current_option_key] += f' {word}'
            elif entity_type == 'ANSWER':
                if current_item is not None:
                    current_item['answer'] += f' {word}'

        elif label == 'O':
            # Accumulate noise word by word
            if noise_buffer:
                noise_buffer += f' {word}'
            else:
                noise_buffer = word

    # Append the last processed item
    if current_item is not None:
        # Note: Trailing 'O' tokens (noise_buffer content) are discarded here.
        structured_data.append(current_item)

    end_time = time.time()
    duration = end_time - start_time

    return structured_data, duration


import os
import re
import base64
import time  # Assuming these imports are available in the scope


def filter_questions(questions: list, image_dir: str = "extracted_images"):
    """
    Filters a list of questions based on option count and content length.
    It removes the 'Q.1' prefix, loads and encodes images from 'noise'/'question'
    to a new 'image' field, and ensures the 'answer' field is set to "null"
    if it is empty.
    """
    start_time = time.time()

    # The robust regex to capture the full tag (Group 1) and the filename base (Group 2).
    # Group 1: The full tag (e.g., '[IMAGE: image_4.png]')
    # Group 2: The filename base (e.g., 'image_4') <--- THIS IS THE KEY FIX
    IMAGE_TAG_PATTERN = r'(\[IMAGE:\s*(image_\d+)\s*\.\s*png\])'

    # NOTE: The IMAGE_DIR definition is moved here to be a default argument for clarity,
    # but the original hardcoded line will also work if imports are at the top.
    # IMAGE_DIR = "extracted_images"

    filtered_questions = []

    for q in questions:
        # 1. Initialize the new 'image' field
        q['image'] = ''
        found_image_base = ''  # Variable to temporarily hold the extracted filename base

        # --- Image Tagging and Q.1 Removal ---

        # Process 'question' field
        if 'question' in q:
            # Remove Q.1 prefix first
            modified_text = re.sub(r'^\s*Q\.\s*1\s*', ' ', q['question'], flags=re.IGNORECASE).strip()
            q['question'] = modified_text

            # Search 'question' field for image tag
            q_match = re.search(IMAGE_TAG_PATTERN, q['question'])
            if q_match:
                # FIX: Extract the filename base from the correct group (Group 2)
                found_image_base = q_match.group(2)
                full_tag = q_match.group(1)

                # Remove the tag from the question text
                q['question'] = re.sub(re.escape(full_tag), ' ', q['question']).strip()

        # Process 'noise' field (takes precedence)
        if 'noise' in q and q.get('noise'):
            noise_match = re.search(IMAGE_TAG_PATTERN, q['noise'])
            if noise_match:
                # FIX: Overwrite the base if found in noise (precedence logic)
                found_image_base = noise_match.group(2)
                full_tag = noise_match.group(1)

                # Remove the tag from the noise text
                q['noise'] = re.sub(re.escape(full_tag), ' ', q['noise']).strip()

        # --- Load and Base64 Encode Image ---
        if found_image_base:
            filename = found_image_base + ".png"
            filepath = os.path.join(image_dir, filename)  # Use the image_dir argument

            if os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as img_file:
                        # Base64 encode the binary data and decode to a string
                        q['image'] = base64.b64encode(img_file.read()).decode("utf-8")
                except IOError as e:
                    # Added robust error handling for file reading
                    print(f"Warning: Could not read image file {filepath}. Error: {e}")
                    q['image'] = ''  # Ensure it remains empty on error
            else:
                print(f"Warning: Image file not found at {filepath}. Base64 field will be empty.")

        # --- Filtering Criteria (Unchanged) ---

        # Modification: If 'answer' field is empty string, set it to "null"
        if q.get('answer') == "":
            q['answer'] = "null"

        options = q.get('options')

        # Criteria 1 & 2: Check options presence and count (2 <= count <= 4)
        is_options_present = isinstance(options, dict) and len(options) > 0
        is_options_count_valid = False
        if is_options_present:
            option_count = len(options)
            is_options_count_valid = (option_count >= 2 and option_count <= 4)

        # Criteria 3: Reject if 2 options, and both are > 3 words
        reject_two_long_options = False
        if is_options_present and len(options) == 2:
            all_options_too_long = True
            for option_text in options.values():
                if len(option_text.split()) <= 3:
                    all_options_too_long = False
                    break
            if all_options_too_long:
                reject_two_long_options = True

        # Append the question only if all criteria are met
        if is_options_present and is_options_count_valid and not reject_two_long_options:
            filtered_questions.append(q)

    end_time = time.time()
    duration = end_time - start_time

    return filtered_questions, duration


# ==============================================================================
# PART 5: INTEGRATION FUNCTION (Main Pipeline)
# ==============================================================================

def process_pdf_pipeline(input_pdf_path: str, model_path: str
                         , unique_id: str = "temp"):
    """
    The main callable function to run the entire pipeline for a given PDF.

    Args:
        input_pdf_path (str): The path to the user's uploaded PDF file.
        model_path (str): Path to the trained LayoutLMv3-CRF model checkpoint.
        unique_id (str): A unique identifier (e.g., a UUID or user_id + timestamp)
                         to name the output files.

    Returns:
        tuple: (raw_bio_predictions: list, final_structured_data: list, annotated_pdf_path: str)
               raw_bio_predictions: The token-level predictions (for "original json").
               final_structured_data: The filtered, clean list of questions/answers.
               annotated_pdf_path: The path to the generated annotated PDF file.
    """
    temp_dir = None
    try:
        # 1. Setup temporary workspace
        pipeline_start_time = time.time()
        temp_dir = tempfile.mkdtemp()
        regenerated_pdf_path = os.path.join(temp_dir, f"{unique_id}_regen.pdf")
        annotated_pdf_path = os.path.join(temp_dir, f"{unique_id}_annotated.pdf")

        print("--- Starting PDF Processing Pipeline ---")
        print(f"Temporary directory: {temp_dir}")

        # 2. Extract and preprocess text
        extracted_text, extraction_duration = extract_text_chunks(input_pdf_path)
        if not extracted_text:
            raise Exception("Extraction failed or returned empty text.")
        print(f"✅ Step 1: Text extraction took {extraction_duration:.2f} seconds.")

        # 3. Create regenerated PDF
        regeneration_duration = create_pdf_from_text_with_markers(extracted_text, output_pdf_path=regenerated_pdf_path)
        print(f"✅ Step 2: PDF regeneration took {regeneration_duration:.2f} seconds.")

        # 4. Run model inference on regenerated PDF
        print("Running LayoutLMv3-CRF inference...")
        raw_bio_predictions, inference_duration, page_times = run_inference_on_pdf(
            regenerated_pdf_path,
            model_path,
            temp_dir,
            annotated_pdf_path
        )
        print(f"✅ Step 3: Total inference took {inference_duration:.2f} seconds.")
        for i, pt in enumerate(page_times):
            print(f"    - Page {i + 1} inference time: {pt:.2f} seconds.")

        # 5. Convert BIO encoding to structured questions/options (Python object)
        intermediate_data, bio_conversion_duration = convert_bio_to_structured_json(raw_bio_predictions)
        print(f"✅ Step 4: BIO to JSON conversion took {bio_conversion_duration:.2f} seconds.")

        # 6. Filter and clean the data
        final_structured_data, filtering_duration = filter_questions(intermediate_data)
        print(f"✅ Step 5: Data filtering took {filtering_duration:.2f} seconds.")

        pipeline_end_time = time.time()
        total_pipeline_duration = pipeline_end_time - pipeline_start_time

        print(f"Total questions extracted: {len(intermediate_data)}")
        print(f"Total valid questions retained: {len(final_structured_data)}")
        print(f"--- Pipeline complete in {total_pipeline_duration:.2f} seconds ---")

        return raw_bio_predictions, final_structured_data, annotated_pdf_path

    except Exception as e:
        print(f"CRITICAL FAILURE IN PIPELINE: {e}")
        # Optionally re-raise the error depending on your application's error handling
        raise
    finally:
        # 7. Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
