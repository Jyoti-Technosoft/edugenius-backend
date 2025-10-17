
import os
import re
import json
import pickle
import time
import base64  # Ensure this is imported
from typing import List, Dict, Any, Tuple
from collections import Counter
from tqdm import tqdm

# --- External Libraries ---
try:
    # PyTorch and related modules
    import torch
    import torch.nn as nn
    from TorchCRF import CRF
    # PDF and Image Processing
    import fitz  # PyMuPDF
    from PIL import Image, ImageEnhance
    import pytesseract
    from fpdf import FPDF as FPDF_BASE
except ImportError as e:
    print(f"Error: Missing library ({e.name}). Please ensure all dependencies are installed.")
    print("Required: pip install PyMuPDF torch torch-crf tqdm Pillow pytesseract fpdf")
    exit()

# ==============================================================================
# 1. CONFIGURATION CONSTANTS (Must match your training/inference script)
# ==============================================================================
DATA_DIR = "output_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BBOX_NORM_CONSTANT = 1000.0
INFERENCE_CHUNK_SIZE = 128
IMAGE_OUTPUT_DIR = "extracted_images"
MAX_CHAR_LEN = 16
EMBED_DIM = 100
CHAR_EMBED_DIM = 30
CHAR_CNN_OUT = 30
BBOX_DIM = 100
HIDDEN_SIZE = 512

LABELS = ["O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION", "B-ANSWER", "I-ANSWER", "B-IMAGE", "I-IMAGE"]
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2IDX = {l: i for l, i in enumerate(LABELS)}

DEFAULT_REGENERATED_PDF = 'temp_standardized_input.pdf'
DEFAULT_BIO_JSON = 'temp_mcq_predictions_bilstm.json'


# ==============================================================================
# 2. CORE UTILITY CLASSES (Vocab, PDF, Model Architecture)
# ==============================================================================

class Vocab:
    """Vocabulary class (Your exact working version)."""

    def __init__(self, min_freq=1, unk_token="<UNK>", pad_token="<PAD>"):
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.freq = Counter()
        self.itos = []
        self.stoi = {}

    def add_sentence(self, toks):
        self.freq.update(toks)

    def build(self):
        items = [tok for tok, c in self.freq.items() if c >= self.min_freq]
        items = [self.pad_token, self.unk_token] + sorted(items)
        self.itos = items
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[self.unk_token])

    def __getstate__(self):
        return {'min_freq': self.min_freq, 'unk_token': self.unk_token, 'pad_token': self.pad_token, 'itos': self.itos,
                'stoi': self.stoi}

    def __setstate__(self, state):
        self.min_freq = state['min_freq']
        self.unk_token = state['unk_token']
        self.pad_token = state['pad_token']
        self.itos = state['itos']
        self.stoi = state['stoi']
        self.freq = Counter()


class PDF(FPDF_BASE):
    """Custom FPDF class with fallback font setup."""

    def __init__(self):
        super().__init__()
        try:
            self.add_font('DejaVuSans', '', 'DejaVuSansCondensed.ttf', uni=True)
            self.set_font('DejaVuSans', '', 12)
        except Exception:
            self.set_font('Arial', '', 12)


class CharCNNEncoder(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, out_dim, kernel_sizes=(3, 4, 5)):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        convs = [nn.Conv1d(char_emb_dim, out_dim, kernel_size=k) for k in kernel_sizes]
        self.convs = nn.ModuleList(convs)
        self.out_dim = out_dim * len(convs)

    def forward(self, char_ids):
        B, L, C = char_ids.size()
        emb = self.char_emb(char_ids.view(B * L, C)).transpose(1, 2)
        outs = [torch.max(torch.relu(conv(emb)), dim=2)[0] for conv in self.convs]
        res = torch.cat(outs, dim=1)
        return res.view(B, L, -1)


class MCQTagger(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, n_labels, bbox_dim=BBOX_DIM):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.char_enc = CharCNNEncoder(char_vocab_size, CHAR_EMBED_DIM, CHAR_CNN_OUT)
        self.bbox_proj = nn.Linear(4, bbox_dim)
        in_dim = EMBED_DIM + self.char_enc.out_dim + bbox_dim
        self.bilstm = nn.LSTM(in_dim, HIDDEN_SIZE // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.ff = nn.Linear(HIDDEN_SIZE, n_labels)
        self.crf = CRF(n_labels)
        self.dropout = nn.Dropout(p=0.5)

    def forward_emissions(self, words, chars, bboxes, mask):
        wemb = self.word_emb(words)
        cenc = self.char_enc(chars)
        benc = self.bbox_proj(bboxes)
        enc_in = torch.cat([wemb, cenc, benc], dim=-1)
        enc_in = self.dropout(enc_in)

        lengths = mask.sum(dim=1).cpu()
        if lengths.max().item() == 0:
            return torch.zeros((enc_in.size(0), enc_in.size(1), len(LABELS)), device=enc_in.device)

        packed_in = nn.utils.rnn.pack_padded_sequence(enc_in, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed_in)
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.ff(padded_out)

    def forward(self, words, chars, bboxes, mask, labels=None, class_weights=None, alpha=0.7):
        emissions = self.forward_emissions(words, chars, bboxes, mask)
        return self.crf.viterbi_decode(emissions, mask=mask)


# ==============================================================================
# 3. HELPER FUNCTIONS (Extraction, Regeneration, and Inference)
# ==============================================================================

def _load_vocabs(path: str) -> Tuple[Vocab, Vocab]:
    """Loads word and character vocabularies."""
    try:
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"Vocab file NOT FOUND at: {absolute_path}")
        with open(absolute_path, "rb") as f:
            word_vocab, char_vocab = pickle.load(f)
        if len(word_vocab) <= 2:
            raise IndexError("CRITICAL: Word vocabulary size is too small. The 'vocabs.pkl' file is invalid.")
        return word_vocab, char_vocab
    except FileNotFoundError:
        raise FileNotFoundError(f"Vocab file not found at {path}. Please train the model first.")
    except Exception as e:
        raise RuntimeError(f"Error loading vocabs from {path}: {e}")


def extract_tokens_from_pdf_pymupdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extracts words and their raw bounding boxes from a PDF using PyMuPDF."""
    all_tokens = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Error opening PDF: {e}")

    for page in doc:
        page_width, page_height = page.rect.width, page.rect.height
        word_list = page.get_text('words', flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for bbox_raw_tuple in word_list:
            word = bbox_raw_tuple[4]
            raw_bbox = [float(bbox_raw_tuple[i]) for i in range(4)]
            normalized_bbox = [
                (raw_bbox[0] / page_width) * BBOX_NORM_CONSTANT,
                (raw_bbox[1] / page_height) * BBOX_NORM_CONSTANT,
                (raw_bbox[2] / page_width) * BBOX_NORM_CONSTANT,
                (raw_bbox[3] / page_height) * BBOX_NORM_CONSTANT
            ]
            all_tokens.append(
                {"word": word, "raw_bbox": raw_bbox, "normalized_bbox": [int(b) for b in normalized_bbox]})
    doc.close()
    return all_tokens


def preprocess_and_collate_tokens(all_tokens: List[Dict[str, Any]], word_vocab: Vocab, char_vocab: Vocab,
                                  chunk_size: int) -> List[Dict[str, Any]]:
    """Chunks the token list, converts to IDs, and prepares batches for inference."""
    all_batches = []
    pad_index = char_vocab.stoi.get(char_vocab.pad_token, 0)

    for i in range(0, len(all_tokens), chunk_size):
        chunk = all_tokens[i:i + chunk_size]
        if not chunk:
            continue

        words = [t["word"] for t in chunk]
        bboxes_norm = [t["normalized_bbox"] for t in chunk]

        word_ids = [word_vocab[w] for w in words]

        char_ids = []
        for w in words:
            chs = [char_vocab[ch] for ch in w[:MAX_CHAR_LEN]]
            if len(chs) < MAX_CHAR_LEN:
                chs += [pad_index] * (MAX_CHAR_LEN - len(chs))
            char_ids.append(chs)

        word_pad = torch.LongTensor([word_ids]).to(DEVICE)
        char_pad = torch.LongTensor([char_ids]).to(DEVICE)

        bbox_pad = torch.FloatTensor([bboxes_norm]).to(DEVICE) / BBOX_NORM_CONSTANT
        mask = torch.ones(word_pad.size(), dtype=torch.bool).to(DEVICE)

        all_batches.append({
            "words": word_pad, "chars": char_pad, "bboxes": bbox_pad, "mask": mask,
            "original_tokens": chunk
        })
    return all_batches


def _run_bilstm_crf_and_save_json(pdf_path: str, model_path: str, vocab_path: str, output_path: str):
    """
    (Stage 3) Runs your exact inference pipeline.
    """
    word_vocab, char_vocab = _load_vocabs(vocab_path)
    model = MCQTagger(len(word_vocab), len(char_vocab), len(LABELS)).to(DEVICE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model checkpoint not found at {model_path}.")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_tokens = extract_tokens_from_pdf_pymupdf(pdf_path)
    batches = preprocess_and_collate_tokens(all_tokens, word_vocab, char_vocab, chunk_size=INFERENCE_CHUNK_SIZE)
    all_predictions = []

    with torch.no_grad():
        for batch in batches:
            words, chars, bboxes, mask = (batch[k] for k in ["words", "chars", "bboxes", "mask"])

            preds_batch = model(words, chars, bboxes, mask)
            predictions = preds_batch[0]

            original_tokens = batch["original_tokens"]

            for token_data, pred_idx in zip(original_tokens, predictions):
                all_predictions.append({
                    "word": token_data["word"],
                    "bbox": [int(b) for b in token_data["raw_bbox"]],
                    "predicted_label": IDX2LABEL[pred_idx]
                })

    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    return True


def _convert_bio_to_structured_json(input_path: str) -> List[Dict[str, Any]]:
    """(Stage 4) Converts raw BIO predictions to a structured list of questions (Fixed Segmentation)."""
    with open(input_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    structured_data = []
    current_item = None
    current_option_key = None
    noise_buffer = ""

    for item in predictions:
        word = item['word']
        label = item['predicted_label']

        if label.startswith('B-'):
            entity_type = label[2:]

            if current_item is not None and entity_type == 'QUESTION':
                # Append the completed question and reset the buffer
                if current_item.get('question', '').strip():
                    current_item['noise'] = current_item.get('noise', '') + " " + noise_buffer.strip()
                    structured_data.append(current_item)
                current_item = None
            # ---------------------------------------------------------------------------------

            noise_buffer = ""  # Clear noise when any 'B-' tag is hit
            if entity_type == 'QUESTION':
                current_item = {'question': word, 'options': {}, 'answer': '', 'noise': ''}
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
            entity_type = label[2:]
            if current_item is not None:
                if entity_type == 'QUESTION':
                    current_item['question'] += f' {word}'
                elif entity_type == 'OPTION' and current_option_key is not None:
                    current_item['options'][current_option_key] += f' {word}'
                elif entity_type == 'ANSWER':
                    current_item['answer'] += f' {word}'
            noise_buffer = ""  # Clear noise when any 'I-' tag is hit

        elif label == 'O':
            # Collect O tokens as potential noise
            noise_buffer += f' {word}' if noise_buffer else word

    # Finalize the last question
    if current_item is not None and current_item.get('question', '').strip():
        current_item['noise'] = current_item.get('noise', '') + " " + noise_buffer.strip()
        structured_data.append(current_item)

    return structured_data



def extract_text_chunks(pdf_path):
    """
    (Stage 1) Your PDF extractor with image saving, tagging, and OCR fallback logic.
    """
    start_time = time.time()
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

        try:
            page_dict = page.get_text("dict", sort=True)
            for block in page_dict.get("blocks", []):
                bbox = block.get("bbox", [0, 0, 0, 0])
                sort_key = (bbox[1], bbox[0])

                if block.get("type") == 0:  # Text block
                    text_block = "".join(
                        "".join(span.get("text", "") for span in line.get("spans", [])) + "\n" for line in
                        block.get("lines", [])).strip()
                    if text_block:
                        if (bbox[1] < HEADER_Y_MAX or bbox[3] > FOOTER_Y_MIN) and re.fullmatch(
                                r'(\d+|[ivx]+|\s*\bPage\s*\d+\b\s*|.{1,15})', text_block, re.IGNORECASE):
                            continue
                        if re.search(r'^[A-D][\.\)]\s', text_block, re.MULTILINE):
                            is_mcq_page = True
                        blocks.append((sort_key, text_block, "text"))

                elif block.get("type") == 1:  # Image block: Extract and Tag
                    try:
                        img_bytes = block.get("image")
                        img_ext = block.get("ext", "png")
                        if not img_bytes:
                            xref = block.get("xref")
                            if xref:
                                img_data = doc.extract_image(xref)
                                img_bytes = img_data.get("image")
                                img_ext = img_data.get("ext", "png")

                        if img_bytes:
                            img_filename = f"image_{global_image_counter}.{img_ext}"
                            img_path = os.path.join(IMAGE_OUTPUT_DIR, img_filename)
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            img_marker = f"[IMAGE: {img_filename}]"
                            blocks.append((sort_key, img_marker, "image"))
                            global_image_counter += 1
                    except Exception as e:
                        print(f"[Error] Critical failure saving image on page {page_num + 1}: {e}")

        except Exception:
            pass


        if len(" ".join(t[1] for t in blocks if t[2] != "image").strip()) < 20 or is_mcq_page:
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert('L')
                img = ImageEnhance.Contrast(img).enhance(3.0)
                img = ImageEnhance.Sharpness(img).enhance(3.0)
                custom_config = r"--psm 6 -l eng"
                if is_mcq_page:
                    custom_config = r"--psm 4 -c preserve_interword_spaces=1"
                ocr_data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)

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
            blocks.sort(key=lambda x: (x[0][0], x[0][1]))
            page_text_parts = []
            current_line_y = -1
            for pos, text, block_type in blocks:
                y = pos[0]
                vertical_gap = abs(y - current_line_y) if current_line_y > 0 else 0
                if current_line_y > 0 and vertical_gap > PARAGRAPH_BREAK_THRESHOLD:
                    page_text_parts.append("[PARAGRAPH_BREAK]")
                elif current_line_y > 0 and vertical_gap > LINE_BREAK_THRESHOLD:
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

    # Global Cleaning and Unicode Fixes
    final_text = final_text.replace("[PARAGRAPH_BREAK]", "\n\n\n")
    final_text = final_text.replace("[BR]", "\n\n\n\n ")
    final_text = re.sub(r'([\.])(\w)', r'\1   \2', final_text)
    final_text = re.sub(r'([\)\]\}:])(\w)', r'\1   \2', final_text)
    final_text = re.sub(r'\n\s*(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\s*\n', '\nQ.1 ', final_text)
    final_text = re.sub(r'\n\s*\d+\s*\n', '\nQ ', final_text)

    #Replace Unicode Smart Quotes and other symbols (PREVENTS FPDF ERROR)
    final_text = final_text.replace('\u2018', "'").replace('\u2019', "'")
    final_text = final_text.replace('\u201c', '"').replace('\u201d', '"')
    final_text = final_text.replace('\u2026', '...')
    final_text = final_text.replace('\u2013', '-')
    final_text = final_text.replace('\u2014', '--')

    final_text = re.sub(r'\n\s*\n\s*\n', '\n\n', final_text).strip()
    return final_text, time.time() - start_time


def create_pdf_from_text_with_markers(text_content, output_pdf_path):
    """
    (Stage 2) Your PDF regeneration logic with dynamic page breaks and header handling.
    """
    start_time = time.time()

    if not text_content:
        raise ValueError("Text content is empty. Cannot create PDF.")

    regex_delimiters = [r'(Q\.\s*\d+)', r'(\n|^)(\d+[\.\)])']
    sections = re.split('|'.join(regex_delimiters), text_content, flags=re.MULTILINE)
    question_blocks = []
    is_first_section = True

    for section in sections:
        if not section or not section.strip():
            continue
        stripped_section = section.strip()
        is_new_question_start = re.match(r'^(Q\.\s*\d+|\d+[\.\)])', stripped_section)
        is_potential_noise = len(stripped_section) < 20

        if is_new_question_start or is_first_section:
            question_blocks.append(stripped_section)
            is_first_section = False
        elif is_potential_noise and question_blocks:
            question_blocks[-1] += ' ' + stripped_section
        else:
            question_blocks[-1] += '\n' + stripped_section

    first_question_block_index = next(
        (idx for idx, block in enumerate(question_blocks) if re.match(r'^(Q\.\s*\d+|\d+[\.\)])', block)), 0)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    questions_on_current_page = 0
    current_block_index = 0
    TIGHT_BREAK_Y = 200
    LOOSE_BREAK_Y = 200

    while current_block_index < len(question_blocks):
        block = question_blocks[current_block_index]
        is_new_question_block = re.match(r'^(Q\.\s*\d+|\d+[\.\)])', block)
        text_to_write = block

        if current_block_index >= first_question_block_index and is_new_question_block:
            questions_on_current_page += 1
            text_to_write = '\n' + block

            if questions_on_current_page == 4 and pdf.get_y() > TIGHT_BREAK_Y:
                pdf.add_page()
                questions_on_current_page = 1
            elif questions_on_current_page == 5 and pdf.get_y() > LOOSE_BREAK_Y:
                pdf.add_page()
                questions_on_current_page = 1
            elif questions_on_current_page > 5:
                pdf.add_page()
                questions_on_current_page = 1

        if text_to_write.strip():
            pdf.multi_cell(0, 5, text=text_to_write)
            pdf.ln(2)

        current_block_index += 1

    pdf.output(output_pdf_path, 'F')
    return time.time() - start_time


def _clean_base64_string(b64_string: str) -> str:
    """
    Aggressively cleans a Base64 string by stripping common malformed characters
    (newlines, whitespace) and ensures correct padding. This addresses the
    'repair' issue you noticed.
    """
    # 1. Strip all whitespace and newline characters
    clean_string = re.sub(r'\s+', '', b64_string)

    # 2. Add padding if missing (Base64 length must be a multiple of 4)
    missing_padding = len(clean_string) % 4
    if missing_padding:
        clean_string += '=' * (4 - missing_padding)

    return clean_string


def _filter_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    (Stage 5) Filters questions, handles images (decodes, removes tag), and cleans text.

    *** FIX APPLIED: Robust Base64 encoding and image tag handling. ***
    """
    # Group 1: Full Tag | Group 2: Filename Base (e.g., 'image_1') | Group 3: Extension (e.g., 'png')
    IMAGE_TAG_PATTERN = r'(\[IMAGE:\s*(image_\d+)\s*\.\s*(png|jpg|jpeg)\s*\])'
    IMAGE_DIR = IMAGE_OUTPUT_DIR
    filtered_questions = []

    for q in questions:
        q['image'] = ''

        # Consolidate image handling and cleaning
        for field in ['question', 'noise']:
            if field in q and q.get(field):

                # 2. Extract image path, encode, and replace marker
                match = re.search(IMAGE_TAG_PATTERN, q[field], re.IGNORECASE)
                if match:
                    full_tag = match.group(1)
                    filename_base = match.group(2)
                    extension = match.group(3)
                    filename = f"{filename_base}.{extension}"
                    filepath = os.path.join(IMAGE_DIR, filename)

                    if os.path.exists(filepath):
                        try:
                            with open(filepath, "rb") as img_file:
                                # Standard Base64 encoding (not URL-safe)
                                encoded_bytes = base64.b64encode(img_file.read())
                                q['image'] = encoded_bytes.decode("utf-8")
                                # No need for aggressive cleaning here, as b64encode is standard

                        except IOError as e:
                            print(f"Warning: Could not read image file {filepath}. Error: {e}")
                            q['image'] = ''
                    else:
                        print(f"Warning: Image file not found at {filepath}. Base64 field will be empty.")

                    # Remove the image tag from the text field AFTER processing
                    q[field] = re.sub(re.escape(full_tag), ' ', q[field]).strip()
                    # Break after finding the first image tag in either question or noise
                    break

                    # 1. Clean question numbers (if any) and general text cleanup
        if q.get('question', ''):
            q['question'] = re.sub(r'^(Q\.\s*\d+\s*|\d+[\.\)]\s*)', ' ', q['question'], flags=re.IGNORECASE).strip()

        # Filtering Logic (Unchanged)
        q['answer'] = q.get('answer') or "null"
        options = q.get('options', {})
        options_count = len(options)

        # Heuristic 1: Options Count (Allowing 2 to 5 options for robustness)
        is_options_count_valid = (2 <= options_count <= 5)

        # Heuristic 2: Reject two long options (To filter out non-MCQ blocks)
        reject_two_long_options = False
        if options_count == 2:
            if all(len(option_text.split()) > 7 for option_text in options.values()):
                reject_two_long_options = True

        if is_options_count_valid and not reject_two_long_options:
            if q.get('question', '').strip():
                filtered_questions.append(q)

    return filtered_questions


def _cleanup_temp_files(paths: List[str]):
    """Removes temporary files after processing."""
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                print(f"Warning: Could not remove temporary file {path}. {e}")


# ==============================================================================
# 4. MAIN INTERFACE FUNCTION (process_pdf_pipeline)
# ==============================================================================

def process_pdf_pipeline(
        input_pdf_path: str,
        model_path: str = os.path.join(DATA_DIR, "model.pt"),
        vocab_path: str = os.path.join(DATA_DIR, "vocabs.pkl"),
        cleanup_temp_files: bool = True
) -> List[Dict[str, Any]]:
    """
    Runs the full BiLSTM-CRF pipeline on a PDF to extract, structure, and filter MCQs.
    """
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"Input PDF '{input_pdf_path}' not found.")

    unique_id = int(time.time() * 1000)
    temp_pdf = os.path.join(os.path.dirname(input_pdf_path) or '.', f"temp_{unique_id}_{DEFAULT_REGENERATED_PDF}")
    temp_bio_json = os.path.join(os.path.dirname(input_pdf_path) or '.', f"temp_{unique_id}_{DEFAULT_BIO_JSON}")
    temp_files = [temp_pdf, temp_bio_json]

    print(f"--- Starting Processing for: {os.path.basename(input_pdf_path)} ---")
    final_questions = []

    try:
        # 1. Preprocessing (Extraction & Cleaning)
        extracted_text, _ = extract_text_chunks(input_pdf_path)
        if not extracted_text.strip():
            print("Extraction failed. Returning empty list.")
            return []

        # 2. Regeneration (Standardizing PDF for consistent tokenization)
        create_pdf_from_text_with_markers(extracted_text, temp_pdf)
        print(f"✅ Stage 1 & 2 Complete: PDF standardized at {temp_pdf}.")

        # 3. Inference (Standardized PDF -> Raw BIO JSON)
        _run_bilstm_crf_and_save_json(temp_pdf, model_path, vocab_path, temp_bio_json)
        print(f"✅ Stage 3 Complete: BIO predictions saved to {temp_bio_json}.")

        # 4. Structuring (Raw BIO JSON -> Structured JSON)
        structured_data = _convert_bio_to_structured_json(temp_bio_json)
        print(f"✅ Stage 4 Complete: {len(structured_data)} questions structured.")

        # 5. Filtering and Final Cleanup (Structured JSON -> Final List)
        final_questions = _filter_questions(structured_data)
        print(f"✅ Stage 5 Complete: {len(final_questions)} valid questions retained.")
        print(f"--- Processing Finished ---")

    except Exception as e:
        print(f"\n--- CRITICAL FAILURE during pipeline execution ---")
        print(f"Error: {type(e).__name__}: {e}")
        raise

    finally:
        if cleanup_temp_files:
            _cleanup_temp_files(temp_files)
            print(" Temporary files cleaned up.")

    return final_questions
