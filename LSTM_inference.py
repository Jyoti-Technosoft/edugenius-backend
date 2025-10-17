
import os
import json
import pickle
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import fitz  # PyMuPDF
import re
from PIL import Image, ImageEnhance
import pytesseract

# ========================================

try:
    # pdfplumber is kept only for completeness if other parts were to rely on it,
    # but the primary extraction logic now uses fitz.
    import pdfplumber
    from TorchCRF import CRF
except ImportError:
    print("Error: Required libraries (fitz/PyMuPDF, pytesseract, and TorchCRF) not found.")
    print("Please install them: pip install PyMuPDF pytesseract torch-crf")
    # Tesseract binary must also be installed on the system path.
    exit()

# ========== CONFIG (Must match Training Script) ==========
DATA_DIR = "output_data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CHAR_LEN = 16
EMBED_DIM = 100
CHAR_EMBED_DIM = 30
CHAR_CNN_OUT = 30
BBOX_DIM = 100
HIDDEN_SIZE = 512
BBOX_NORM_CONSTANT = 1000.0
INFERENCE_CHUNK_SIZE = 256

# ========== LABELS (Must match Training Script) ==========
LABELS = ["O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION", "B-ANSWER", "I-ANSWER", "B-IMAGE", "I-IMAGE"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}


class Vocab:
    def __init__(self, min_freq=1, unk_token="<UNK>", pad_token="<PAD>"):
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.freq = Counter()
        self.itos = []  # Index to String
        self.stoi = {}  # String to Index

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
        """Allows lookup using word_vocab[token]. Returns UNK index if token is not found."""
        return self.stoi.get(token, self.stoi[self.unk_token])

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
    """Loads word and character vocabularies from a pickle file and verifies size."""
    try:
        absolute_path = os.path.abspath(path)
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"Vocab file NOT FOUND at: {absolute_path}")
        with open(absolute_path, "rb") as f:
            word_vocab, char_vocab = pickle.load(f)
        if len(word_vocab) <= 2:
            raise IndexError("CRITICAL: Word vocabulary size is too small. Vocab file is invalid.")
        return word_vocab, char_vocab
    except FileNotFoundError:
        raise FileNotFoundError(f"Vocab file not found at {path}. Please run the training script first.")
    except Exception as e:
        raise RuntimeError(f"Error loading vocabs from {path}: {e}")


# -------------------------
# Model Architecture (Unchanged)
# -------------------------

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
            B, L = enc_in.size(0), enc_in.size(1)
            return torch.zeros((B, L, len(LABELS)), device=enc_in.device)

        packed_in = nn.utils.rnn.pack_padded_sequence(enc_in, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed_in)
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        return self.ff(padded_out)

    def forward(self, words, chars, bboxes, mask, labels=None, class_weights=None, alpha=0.7):
        emissions = self.forward_emissions(words, chars, bboxes, mask)
        return self.crf.viterbi_decode(emissions, mask=mask)


# -------------------------
# PDF Processing with OCR Fallback
# -------------------------

# OCR FALLBACK: Now takes raw page and dimensions, as it's used only when text extraction fails.
def ocr_fallback_page(page: fitz.Page, page_width: float, page_height: float) -> List[Dict[str, Any]]:
    """
    Renders a PyMuPDF page, runs Tesseract OCR, and tokenizes the result.
    """
    try:
        # Render page at high resolution (300 DPI equivalent)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        if pix.n - pix.alpha > 3:  # Handle CMYK
            pix = fitz.Pixmap(fitz.csRGB, pix)

        img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Preprocessing for Tesseract (as was in the original code)
        img_pil = img_pil.convert('L')
        img_pil = ImageEnhance.Contrast(img_pil).enhance(2.0)
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(2.0)

        # Run Tesseract
        ocr_data = pytesseract.image_to_data(img_pil, output_type=pytesseract.Output.DICT)

        ocr_tokens = []
        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i]
            conf = ocr_data['conf'][i]

            # Use only words with reasonable confidence
            if word.strip() and int(conf) > 50:
                # Get Tesseract's raw pixel bounding box
                left = ocr_data['left'][i]
                top = ocr_data['top'][i]
                width = ocr_data['width'][i]
                height = ocr_data['height'][i]

                # Convert pixel bbox back to original PDF coordinate system
                scale = page_width / pix.width

                raw_bbox = [
                    left * scale,
                    top * scale,
                    (left + width) * scale,
                    (top + height) * scale
                ]

                # Normalize bbox
                normalized_bbox = [
                    (raw_bbox[0] / page_width) * BBOX_NORM_CONSTANT,
                    (raw_bbox[1] / page_height) * BBOX_NORM_CONSTANT,
                    (raw_bbox[2] / page_width) * BBOX_NORM_CONSTANT,
                    (raw_bbox[3] / page_height) * BBOX_NORM_CONSTANT
                ]

                ocr_tokens.append({
                    "word": word,
                    "raw_bbox": [int(b) for b in raw_bbox],
                    "normalized_bbox": [int(b) for b in normalized_bbox]
                })

        return ocr_tokens

    except Exception as e:
        # Note: 'page.number' might not be available if not running in a loop context
        print(f"OCR fallback failed: {e}")
        return []


# EXTRACTION FUNCTION: Prioritizes PyMuPDF text layer
def extract_tokens_from_pdf_fitz_with_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts words and their raw bounding boxes using PyMuPDF (fitz) text layer
    and falls back to OCR if no text is found.
    """
    all_tokens = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in tqdm(range(len(doc)), desc="PDF Page Processing"):
            page = doc.load_page(page_num)
            page_width, page_height = page.rect.width, page.rect.height
            page_tokens = []

            # 1. Primary Extraction: Use PyMuPDF's word structure (fitz.Page.get_text("words"))
            # word_list format: (x0, y0, x1, y1, word, ...)
            word_list = page.get_text("words", sort=True)

            if word_list:
                for word_data in word_list:
                    word = word_data[4]
                    raw_bbox = word_data[:4]

                    # Normalize bboxes
                    normalized_bbox = [
                        (raw_bbox[0] / page_width) * BBOX_NORM_CONSTANT,
                        (raw_bbox[1] / page_height) * BBOX_NORM_CONSTANT,
                        (raw_bbox[2] / page_width) * BBOX_NORM_CONSTANT,
                        (raw_bbox[3] / page_height) * BBOX_NORM_CONSTANT
                    ]

                    page_tokens.append({
                        "word": word,
                        "raw_bbox": [int(b) for b in raw_bbox],
                        "normalized_bbox": [int(b) for b in normalized_bbox]
                    })

            # 2. OCR Fallback
            if not page_tokens:
                print(f" (Page {page_num + 1}) No text layer found. Running OCR...")
                page_tokens = ocr_fallback_page(page, page_width, page_height)

            all_tokens.extend(page_tokens)

        doc.close()
    except Exception as e:
        raise RuntimeError(f"Error opening or processing PDF with fitz/OCR: {e}")

    return all_tokens


# NOTE: The main execution pointer already uses the robust function name
extract_tokens_from_pdf = extract_tokens_from_pdf_fitz_with_ocr


def preprocess_and_collate_tokens(all_tokens: List[Dict[str, Any]], word_vocab: Vocab, char_vocab: Vocab,
                                  chunk_size: int) -> List[Dict[str, Any]]:
    """
    Chunks the token list, converts to IDs, and prepares batches for inference. (Unchanged)
    """
    all_batches = []

    for i in range(0, len(all_tokens), chunk_size):
        chunk = all_tokens[i:i + chunk_size]
        if not chunk: continue

        words = [t["word"] for t in chunk]
        bboxes_norm = [t["normalized_bbox"] for t in chunk]

        # Convert to IDs
        word_ids = [word_vocab[w] for w in words]

        char_ids = []
        for w in words:
            chs = [char_vocab[ch] for ch in w[:MAX_CHAR_LEN]]
            if len(chs) < MAX_CHAR_LEN:
                pad_index = char_vocab.stoi.get(char_vocab.pad_token, 0)
                chs += [pad_index] * (MAX_CHAR_LEN - len(chs))
            char_ids.append(chs)

        # Create padded tensors (using single-sample batches)
        word_pad = torch.LongTensor([word_ids]).to(DEVICE)
        char_pad = torch.LongTensor([char_ids]).to(DEVICE)

        # Final normalization to [0, 1] range before feeding to the model
        bbox_pad = torch.FloatTensor([bboxes_norm]).to(DEVICE) / BBOX_NORM_CONSTANT
        mask = torch.ones(word_pad.size(), dtype=torch.bool).to(DEVICE)

        all_batches.append({
            "words": word_pad,
            "chars": char_pad,
            "bboxes": bbox_pad,
            "mask": mask,
            "original_tokens": chunk  # Keep the original data for output formatting
        })

    return all_batches


# -------------------------
# Main Inference Logic 
# -------------------------

def run_inference(pdf_path: str, model_path: str, vocab_path: str, output_path: str):
    """
    Main function to orchestrate loading, processing, inference, and output.
    """
    print(f"Loading vocabs from {vocab_path}...")
    try:
        word_vocab, char_vocab = load_vocabs(vocab_path)
    except Exception as e:
        print(f"\n❌ FATAL ERROR LOADING VOCABS: {e}")
        return

    print(f"Loading model architecture and weights from {model_path}...")
    model = MCQTagger(len(word_vocab), len(char_vocab), len(LABELS)).to(DEVICE)
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at {model_path}. Train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully.")
    print(f"Extracting tokens and bboxes from {pdf_path} using PyMuPDF/OCR...")
    try:
        all_tokens = extract_tokens_from_pdf(pdf_path)
    except RuntimeError as e:
        print(f"❌ PDF Processing Error: {e}")
        return

    if not all_tokens:
        print("❌ ERROR: No tokens were extracted from the PDF, even after OCR fallback.")
        return

    print(f"Total tokens extracted: {len(all_tokens)}. Preparing batches...")

    batches = preprocess_and_collate_tokens(all_tokens, word_vocab, char_vocab, chunk_size=INFERENCE_CHUNK_SIZE)

    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(batches, desc="Running Inference"):
            words, chars, bboxes, mask = (batch[k] for k in ["words", "chars", "bboxes", "mask"])

            preds_batch = model(words, chars, bboxes, mask)
            predictions = preds_batch[0]

            original_tokens = batch["original_tokens"]

            for token_data, pred_idx in zip(original_tokens, predictions):
                all_predictions.append({
                    "word": token_data["word"],
                    "bbox": token_data["raw_bbox"],  # Use the raw_bbox (which is int-list)
                    "predicted_label": IDX2LABEL[pred_idx]
                })

    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Inference complete. Total tokens predicted: {len(all_predictions)}")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bi-LSTM-CRF Inference Script for PDF to BIO-JSON conversion.")
    parser.add_argument("--input_pdf", type=str, required=True,
                        help="Path to the input PDF file for inference.")
    parser.add_argument("--model_path", type=str, default=os.path.join(DATA_DIR, "model_CAT.pt"),
                        help="Path to the saved PyTorch model checkpoint.")
    parser.add_argument("--vocab_path", type=str, default=os.path.join(DATA_DIR, "vocabs_CAT.pkl"),
                        help="Path to the saved vocabulary pickle file.")
    parser.add_argument("--output_json", type=str, default="mcq_predictions_bilstm.json",
                        help="Path to save the output BIO-encoded JSON predictions.")

    args = parser.parse_args()

    run_inference(args.input_pdf, args.model_path, args.vocab_path, args.output_json)
