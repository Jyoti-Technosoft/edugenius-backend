
import fitz  # PyMuPDF
import numpy as np
import cv2
import torch
import torch
import torch.serialization

_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # FORCE classic behavior
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load





#==================================================================================
#RAPID OCR
#==================================================================================

from rapidocr import RapidOCR, OCRVersion

# Initialize RapidOCR (v5 is generally the most accurate current version)
# We use return_word_box=True to get word-level precision similar to Tesseract's image_to_data
ocr_engine = RapidOCR(params={
    "Det.ocr_version": OCRVersion.PPOCRV5,
    "Rec.ocr_version": OCRVersion.PPOCRV5,
    "Cls.ocr_version": OCRVersion.PPOCRV4,
})



#==================================================================================
#RAPID OCR
#==================================================================================



import json
import argparse
import os
import re

import torch.nn as nn
from TorchCRF import CRF
# from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Model, LayoutLMv3Config
from transformers import LayoutLMv3Tokenizer, LayoutLMv3Model, LayoutLMv3Config
from typing import List, Dict, Any, Optional, Union, Tuple
from ultralytics import YOLO
import glob
import pytesseract
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import sys
import io
import base64
import tempfile
import time
import shutil
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq



# ============================================================================
# --- TR-OCR/ORT MODEL INITIALIZATION ---
# ============================================================================

logging.basicConfig(level=logging.WARNING)

processor = None
ort_model = None

try:
    MODEL_NAME = 'breezedeus/pix2text-mfr-1.5'
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    
    # Initialize the model for ONNX Runtime
    # NOTE: Set use_cache=False to avoid caching warnings/issues if reloading
    ort_model = ORTModelForVision2Seq.from_pretrained(MODEL_NAME, use_cache=False)
    
    print("✅ ORTModelForVision2Seq and TrOCRProcessor initialized successfully for equation conversion.")
except Exception as e:
    print(f"❌ Error initializing TrOCR/ORT model. Equations will not be converted: {e}")
    processor = None
    ort_model = None



#=====================================================================================================================
#=====================================================================================================================



# ============================================================================
# --- CUSTOM MODEL DEFINITIONS (ADD THIS BLOCK) ---
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import pickle

# --- CONSTANTS FOR CUSTOM MODEL ---
MODEL_FILE = "model_enhanced.pt"      # Ensure this file is in your directory
VOCAB_FILE = "vocabs_enhanced.pkl"    # Ensure this file is in your directory
DEVICE = torch.device("cpu")          # Use "cuda" if available
MAX_CHAR_LEN = 16
EMBED_DIM = 128
CHAR_EMBED_DIM = 50
CHAR_CNN_OUT = 50
BBOX_DIM = 128
HIDDEN_SIZE = 768
SPATIAL_FEATURE_DIM = 64
POSITIONAL_DIM = 128
INFERENCE_CHUNK_SIZE = 450

LABELS = [
    "O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION", 
    "B-ANSWER", "I-ANSWER", "B-IMAGE", "I-IMAGE", 
    "B-SECTION HEADING", "I-SECTION HEADING", "B-PASSAGE", "I-PASSAGE"
]
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}

# --- CRF DEPENDENCY ---
try:
    from torch_crf import CRF
except ImportError:
    try:
        from TorchCRF import CRF
    except ImportError:
        # Minimal fallback if CRF library is missing (though you should install it)
        class CRF(nn.Module):
            def __init__(self, *args, **kwargs): super().__init__()

# --- MODEL CLASSES ---
class Vocab:
    def __init__(self, min_freq=1, unk_token="<UNK>", pad_token="<PAD>"):
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.freq = Counter()
        self.itos = []
        self.stoi = {}
    def __len__(self): return len(self.itos)
    def __getitem__(self, token): return self.stoi.get(token, self.stoi.get(self.unk_token, 0))

class CharCNNEncoder(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, out_dim, kernel_sizes=(2, 3, 4, 5)):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(char_emb_dim, out_dim, kernel_size=k) for k in kernel_sizes])
        self.out_dim = out_dim * len(kernel_sizes)
    def forward(self, char_ids):
        B, L, C = char_ids.size()
        emb = self.char_emb(char_ids.view(B * L, C)).transpose(1, 2)
        outs = [torch.max(torch.relu(conv(emb)), dim=2)[0] for conv in self.convs]
        return torch.cat(outs, dim=1).view(B, L, -1)

class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
    def forward(self, x, mask):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        mask_expanded = mask.unsqueeze(1).expand_as(scores)
        scores = scores.masked_fill(~mask_expanded, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1).masked_fill(torch.isnan(scores), 0.0)
        return torch.matmul(attn_weights, V)

class MCQTagger(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, n_labels):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.char_enc = CharCNNEncoder(char_vocab_size, CHAR_EMBED_DIM, CHAR_CNN_OUT)
        self.bbox_proj = nn.Sequential(nn.Linear(4, BBOX_DIM), nn.ReLU(), nn.Dropout(0.1), nn.Linear(BBOX_DIM, BBOX_DIM))
        self.spatial_proj = nn.Sequential(nn.Linear(11, SPATIAL_FEATURE_DIM), nn.ReLU(), nn.Dropout(0.1))
        self.context_proj = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Dropout(0.1))
        self.positional_encoding = nn.Embedding(512, POSITIONAL_DIM)
        
        in_dim = (EMBED_DIM + self.char_enc.out_dim + BBOX_DIM + SPATIAL_FEATURE_DIM + 32 + POSITIONAL_DIM)
        
        self.bilstm = nn.LSTM(in_dim, HIDDEN_SIZE // 2, num_layers=3, batch_first=True, bidirectional=True, dropout=0.3)
        self.spatial_attention = SpatialAttention(HIDDEN_SIZE)
        
        # --- FIX: ADD THIS LINE TO MATCH SAVED MODEL ---
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE) 
        # -----------------------------------------------

        self.ff = nn.Sequential(nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE), nn.ReLU(), nn.Dropout(0.3), nn.Linear(HIDDEN_SIZE, n_labels))
        self.crf = CRF(n_labels)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, words, chars, bboxes, spatial_feats, context_feats, mask):
        B, L = words.size()
        wemb = self.word_emb(words)
        cenc = self.char_enc(chars)
        benc = self.bbox_proj(bboxes)
        senc = self.spatial_proj(spatial_feats)
        cxt_enc = self.context_proj(context_feats)
        
        pos = torch.arange(L, device=words.device).unsqueeze(0).expand(B, -1)
        pos_enc = self.positional_encoding(pos.clamp(max=511))
        
        enc_in = self.dropout(torch.cat([wemb, cenc, benc, senc, cxt_enc, pos_enc], dim=-1))
        
        lengths = mask.sum(dim=1).cpu()
        packed_in = nn.utils.rnn.pack_padded_sequence(enc_in, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed_in)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        attn_out = self.spatial_attention(lstm_out, mask)
        
        # Note: Even if layer_norm isn't explicitly used in the forward pass logic here,
        # it must be defined in __init__ to satisfy the strict state_dict loading.
        
        emissions = self.ff(torch.cat([lstm_out, attn_out], dim=-1))
        return self.crf.viterbi_decode(emissions, mask=mask)
# --- INJECT DEPENDENCIES FOR PICKLE LOADING ---
import sys
from types import ModuleType
train_mod = ModuleType("train_model")
sys.modules["train_model"] = train_mod
train_mod.Vocab = Vocab
train_mod.MCQTagger = MCQTagger
train_mod.CharCNNEncoder = CharCNNEncoder
train_mod.SpatialAttention = SpatialAttention






# ============================================================================
# --- CUSTOM FEATURE EXTRACTORS ---
# ============================================================================
def extract_spatial_features(tokens, idx):
    curr = tokens[idx]
    f = []
    # Vertical distance to next
    if idx < len(tokens)-1: f.append(min((tokens[idx+1]['y0'] - curr['y1'])/100.0, 1.0))
    else: f.append(0.0)
    # Vertical distance from prev
    if idx > 0: f.append(min((curr['y0'] - tokens[idx-1]['y1'])/100.0, 1.0))
    else: f.append(0.0)
    # Geometry
    f.extend([curr['x0']/1000.0, (curr['x1']-curr['x0'])/1000.0, (curr['y1']-curr['y0'])/1000.0])
    f.extend([(curr['x0']+curr['x1'])/2000.0, (curr['y0']+curr['y1'])/2000.0, curr['x0']/1000.0])
    # Aspect ratio
    f.append(min(((curr['x1']-curr['x0'])/max((curr['y1']-curr['y0']),1.0))/10.0, 1.0))
    # Alignment check
    if idx > 0: f.append(float(abs(curr['x0'] - tokens[idx-1]['x0']) < 5))
    else: f.append(0.0)
    # Area
    f.append(min(((curr['x1']-curr['x0'])*(curr['y1']-curr['y0']))/(1000.0**2), 1.0))
    return f

def extract_context_features(tokens, idx, window=3):
    f = []
    def check_p(i):
        t = str(tokens[i]['word']).lower().strip() # Changed 'text' to 'word' to match pipeline
        return [float(bool(re.match(r'^q?\.?\d+[.:]', t))), float(bool(re.match(r'^[a-dA-D][.)]', t))), float(t.isupper() and len(t)>2)]
    
    prev_res = [0.0, 0.0, 0.0]
    for i in range(max(0, idx-window), idx):
        res = check_p(i)
        prev_res = [max(prev_res[j], res[j]) for j in range(3)]
    f.extend(prev_res)
    next_res = [0.0, 0.0, 0.0]
    for i in range(idx+1, min(len(tokens), idx+window+1)):
        res = check_p(i)
        next_res = [max(next_res[j], res[j]) for j in range(3)]
    f.extend(next_res)
    dq, dopt = 1.0, 1.0
    for i in range(idx+1, min(len(tokens), idx+window+1)):
        t = str(tokens[i]['word']).lower().strip()
        if re.match(r'^q?\.?\d+[.:]', t): dq = min(dq, (i-idx)/window)
        if re.match(r'^[a-dA-D][.)]', t): dopt = min(dopt, (i-idx)/window)
    f.extend([dq, dopt])
    return f

#======================================================================================================================================================
#======================================================================================================================================================

from typing import Optional

def sanitize_text(text: Optional[str]) -> str:
    """Removes surrogate characters and other invalid code points that cause UTF-8 encoding errors."""
    if not isinstance(text, str) or text is None:
        return ""
    
    # Matches all surrogates (\ud800-\udfff) and common non-characters (\ufffe, \uffff).
    # This specifically removes '\udefd' which is causing your error.
    surrogates_and_nonchars = re.compile(r'[\ud800-\udfff\ufffe\uffff]')
    
    # Replace the invalid characters with a standard space.
    # We strip afterward in the calling function.
    return surrogates_and_nonchars.sub(' ', text)





def get_latex_from_base64(base64_string: str) -> str:
    """
    Decodes a Base64 image string and uses the pre-initialized TrOCR/ORT model
    to recognize the formula. It cleans the output by removing spaces and
    crucially, replacing double backslashes with single backslashes for correct LaTeX.
    """
    if ort_model is None or processor is None:
        return "[MODEL_ERROR: Model not initialized]"

    try:
        # 1. Decode Base64 to Image
        image_data = base64.b64decode(base64_string)
        # We must ensure the image is RGB format for the model input
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 2. Preprocess the image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        # 3. Text Generation (OCR)
        generated_ids = ort_model.generate(pixel_values)
        raw_generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        if not raw_generated_text:
            return "[OCR_WARNING: No formula found]"
            
        latex_string = raw_generated_text[0]
        
        # --- 4. Post-processing and Cleanup ---
        
        # # A. Remove all spaces/line breaks
        # cleaned_latex = re.sub(r'\s+', '', latex_string)
        cleaned_latex = re.sub(r'[\r\n]+', '', latex_string)
        
        # B. CRITICAL FIX: Replace double backslashes (\\) with single backslashes (\).
        # This corrects model output that already over-escaped the LaTeX commands.
        # Python literal: '\\\\' is replaced with '\\'.
        #cleaned_latex = cleaned_latex.replace('\\\\', '\\')
      
        return cleaned_latex


    except Exception as e:
        # Catch any unexpected errors
        print(f"  ❌ TR-OCR Recognition failed: {e}")
        return f"[TR_OCR_ERROR: Recognition failed: {e}]"







# ============================================================================
# --- CONFIGURATION AND CONSTANTS ---
# ============================================================================


# NOTE: Update these paths to match your environment before running!
WEIGHTS_PATH = 'best.pt'
DEFAULT_LAYOUTLMV3_MODEL_PATH = "98.pth"

# DIRECTORY CONFIGURATION
OCR_JSON_OUTPUT_DIR = './ocr_json_output_final'
FIGURE_EXTRACTION_DIR = './figure_extraction'
TEMP_IMAGE_DIR = './temp_pdf_images'

# Detection parameters
# CONF_THRESHOLD = 0.2
TARGET_CLASSES = ['figure', 'equation']
IOU_MERGE_THRESHOLD = 0.4
IOA_SUPPRESSION_THRESHOLD = 0.7
LINE_TOLERANCE = 15

# Similarity
SIMILARITY_THRESHOLD = 0.10
RESOLUTION_MARGIN = 0.05

# Global counters for sequential numbering across the entire PDF
GLOBAL_FIGURE_COUNT = 0
GLOBAL_EQUATION_COUNT = 0

# LayoutLMv3 Labels
ID_TO_LABEL = {
    0: "O",
    1: "B-QUESTION", 2: "I-QUESTION",
    3: "B-OPTION", 4: "I-OPTION",
    5: "B-ANSWER", 6: "I-ANSWER",
    7: "B-SECTION_HEADING", 8: "I-SECTION_HEADING",
    9: "B-PASSAGE", 10: "I-PASSAGE"
}
NUM_LABELS = len(ID_TO_LABEL)


# ============================================================================
# --- PERFORMANCE OPTIMIZATION: OCR CACHE ---
# ============================================================================

class OCRCache:
    """Caches OCR results per page to avoid redundant Tesseract runs."""

    def __init__(self):
        self.cache = {}

    def get_key(self, pdf_path: str, page_num: int) -> str:
        return f"{pdf_path}:{page_num}"

    def has_ocr(self, pdf_path: str, page_num: int) -> bool:
        return self.get_key(pdf_path, page_num) in self.cache

    def get_ocr(self, pdf_path: str, page_num: int) -> Optional[list]:
        return self.cache.get(self.get_key(pdf_path, page_num))

    def set_ocr(self, pdf_path: str, page_num: int, ocr_data: list):
        self.cache[self.get_key(pdf_path, page_num)] = ocr_data

    def clear(self):
        self.cache.clear()


# Global OCR cache instance
_ocr_cache = OCRCache()


# ============================================================================
# --- PHASE 1: YOLO/OCR PREPROCESSING FUNCTIONS ---
# ============================================================================

def calculate_iou(box1, box2):
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box_a_area = (x2_a - x1_a) * (y2_a - y1_a)
    box_b_area = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = float(box_a_area + box_b_area - intersection_area)
    return intersection_area / union_area if union_area > 0 else 0


def calculate_ioa(box1, box2):
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box_a_area = (x2_a - x1_a) * (y2_a - y1_a)
    return intersection_area / box_a_area if box_a_area > 0 else 0


def filter_nested_boxes(detections, ioa_threshold=0.80):
    """
    Removes boxes that are inside larger boxes (Containment Check).
    Prioritizes keeping the LARGEST box (the 'parent' container).
    """
    if not detections:
        return []

    # 1. Calculate Area for all detections
    for d in detections:
        x1, y1, x2, y2 = d['coords']
        d['area'] = (x2 - x1) * (y2 - y1)

    # 2. Sort by Area Descending (Largest to Smallest)
    # This ensures we process the 'container' first
    detections.sort(key=lambda x: x['area'], reverse=True)

    keep_indices = []
    is_suppressed = [False] * len(detections)

    for i in range(len(detections)):
        if is_suppressed[i]: continue

        keep_indices.append(i)
        box_a = detections[i]['coords']

        # Compare with all smaller boxes
        for j in range(i + 1, len(detections)):
            if is_suppressed[j]: continue

            box_b = detections[j]['coords']

            # Calculate Intersection
            x_left = max(box_a[0], box_b[0])
            y_top = max(box_a[1], box_b[1])
            x_right = min(box_a[2], box_b[2])
            y_bottom = min(box_a[3], box_b[3])

            if x_right < x_left or y_bottom < y_top:
                intersection = 0
            else:
                intersection = (x_right - x_left) * (y_bottom - y_top)

            # Calculate IoA (Intersection over Area of the SMALLER box)
            area_b = detections[j]['area']

            if area_b > 0:
                ioa_small = intersection / area_b

                # If the small box is > 90% inside the big box, suppress the small one.
                if ioa_small > ioa_threshold:
                    is_suppressed[j] = True
                    # print(f"    [Suppress] Removed nested object inside larger '{detections[i]['class']}'")

    return [detections[i] for i in keep_indices]


def merge_overlapping_boxes(detections, iou_threshold):
    if not detections: return []
    detections.sort(key=lambda d: d['conf'], reverse=True)
    merged_detections = []
    is_merged = [False] * len(detections)
    for i in range(len(detections)):
        if is_merged[i]: continue
        current_box = detections[i]['coords']
        current_class = detections[i]['class']
        merged_x1, merged_y1, merged_x2, merged_y2 = current_box
        for j in range(i + 1, len(detections)):
            if is_merged[j] or detections[j]['class'] != current_class: continue
            other_box = detections[j]['coords']
            iou = calculate_iou(current_box, other_box)
            if iou > iou_threshold:
                merged_x1 = min(merged_x1, other_box[0])
                merged_y1 = min(merged_y1, other_box[1])
                merged_x2 = max(merged_x2, other_box[2])
                merged_y2 = max(merged_y2, other_box[3])
                is_merged[j] = True
        merged_detections.append({
            'coords': (merged_x1, merged_y1, merged_x2, merged_y2),
            'y1': merged_y1, 'class': current_class, 'conf': detections[i]['conf']
        })
    return merged_detections


def merge_yolo_into_word_data(raw_word_data: list, yolo_detections: list, scale_factor: float) -> list:
    """
    Filters out raw words that are inside YOLO boxes and replaces them with
    a single solid 'placeholder' block for the column detector.
    """
    if not yolo_detections:
        return raw_word_data

    # 1. Convert YOLO boxes (Pixels) to PDF Coordinates (Points)
    pdf_space_boxes = []
    for det in yolo_detections:
        x1, y1, x2, y2 = det['coords']
        pdf_box = (
            x1 / scale_factor,
            y1 / scale_factor,
            x2 / scale_factor,
            y2 / scale_factor
        )
        pdf_space_boxes.append(pdf_box)

    # 2. Filter out raw words that are inside YOLO boxes
    cleaned_word_data = []
    for word_tuple in raw_word_data:
        wx1, wy1, wx2, wy2 = word_tuple[1], word_tuple[2], word_tuple[3], word_tuple[4]
        w_center_x = (wx1 + wx2) / 2
        w_center_y = (wy1 + wy2) / 2

        is_inside_yolo = False
        for px1, py1, px2, py2 in pdf_space_boxes:
            if px1 <= w_center_x <= px2 and py1 <= w_center_y <= py2:
                is_inside_yolo = True
                break

        if not is_inside_yolo:
            cleaned_word_data.append(word_tuple)

    # 3. Add the YOLO boxes themselves as "Solid Words"
    for i, (px1, py1, px2, py2) in enumerate(pdf_space_boxes):
        dummy_entry = (f"BLOCK_{i}", px1, py1, px2, py2)
        cleaned_word_data.append(dummy_entry)

    return cleaned_word_data


# ============================================================================
# --- MISSING HELPER FUNCTION ---
# ============================================================================

def preprocess_image_for_ocr(img_np):
    """
    Converts image to grayscale and applies Otsu's Binarization
    to separate text from background clearly.
    """
    # 1. Convert to Grayscale if needed
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np

    # 2. Apply Otsu's Thresholding (Automatic binary threshold)
    # This makes text solid black and background solid white
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def calculate_vertical_gap_coverage(word_data: list, sep_x: int, page_height: float, gutter_width: int = 10) -> float:
    """
    Calculates what percentage of the page's vertical text span is 'cleanly split' by the separator.
    A valid column split should split > 65% of the page verticality.
    """
    if not word_data:
        return 0.0

    # Determine the vertical span of the actual text content
    y_coords = [w[2] for w in word_data] + [w[4] for w in word_data]  # y1 and y2
    min_y, max_y = min(y_coords), max(y_coords)
    total_text_height = max_y - min_y

    if total_text_height <= 0:
        return 0.0

    # Create a boolean array representing the Y-axis (1 pixel per unit)
    gap_open_mask = np.ones(int(total_text_height) + 1, dtype=bool)

    zone_left = sep_x - (gutter_width / 2)
    zone_right = sep_x + (gutter_width / 2)
    offset_y = int(min_y)

    for _, x1, y1, x2, y2 in word_data:
        # Check if this word horizontally interferes with the separator
        if x2 > zone_left and x1 < zone_right:
            y_start_idx = max(0, int(y1) - offset_y)
            y_end_idx = min(len(gap_open_mask), int(y2) - offset_y)
            if y_end_idx > y_start_idx:
                gap_open_mask[y_start_idx:y_end_idx] = False

    open_pixels = np.sum(gap_open_mask)
    coverage_ratio = open_pixels / len(gap_open_mask)

    return coverage_ratio


def calculate_x_gutters(word_data: list, params: Dict, page_height: float) -> List[int]:
    """
    Calculates X-axis histogram and validates using BRIDGING DENSITY and Vertical Coverage.
    """
    if not word_data: return []

    x_points = []
    # Use only word_data elements 1 (x1) and 3 (x2)
    for item in word_data:
        x_points.extend([item[1], item[3]])

    if not x_points: return []
    max_x = max(x_points)

    # 1. Determine total text height for ratio calculation
    y_coords = [item[2] for item in word_data] + [item[4] for item in word_data]
    min_y, max_y = min(y_coords), max(y_coords)
    total_text_height = max_y - min_y
    if total_text_height <= 0: return []

    # Histogram Setup
    bin_size = params.get('cluster_bin_size', 5)
    smoothing = params.get('cluster_smoothing', 1)
    min_width = params.get('cluster_min_width', 20)
    threshold_percentile = params.get('cluster_threshold_percentile', 85)

    num_bins = int(np.ceil(max_x / bin_size))
    hist, bin_edges = np.histogram(x_points, bins=num_bins, range=(0, max_x))
    smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=smoothing)
    inverted_signal = np.max(smoothed_hist) - smoothed_hist

    peaks, properties = find_peaks(
        inverted_signal,
        height=np.max(inverted_signal) - np.percentile(smoothed_hist, threshold_percentile),
        distance=min_width / bin_size
    )

    if not peaks.size: return []
    separator_x_coords = [int(bin_edges[p]) for p in peaks]
    final_separators = []

    for x_coord in separator_x_coords:
        # --- CHECK 1: BRIDGING DENSITY (The "Cut Through" Check) ---
        # Calculate the total vertical height of words that physically cross this line.
        bridging_height = 0
        bridging_count = 0

        for item in word_data:
            wx1, wy1, wx2, wy2 = item[1], item[2], item[3], item[4]

            # Check if this word physically sits on top of the separator line
            if wx1 < x_coord and wx2 > x_coord:
                word_h = wy2 - wy1
                bridging_height += word_h
                bridging_count += 1

        # Calculate Ratio: How much of the page's text height is blocked by these crossing words?
        bridging_ratio = bridging_height / total_text_height

        # THRESHOLD: If bridging blocks > 8% of page height, REJECT.
        # This allows for page numbers or headers (usually < 5%) to cross, but NOT paragraphs.
        if bridging_ratio > 0.08:
            print(
                f"      ❌ Separator X={x_coord} REJECTED: Bridging Ratio {bridging_ratio:.1%} (>15%) cuts through text.")
            continue

        # --- CHECK 2: VERTICAL GAP COVERAGE (The "Clean Split" Check) ---
        # The gap must exist cleanly for > 65% of the text height.
        coverage = calculate_vertical_gap_coverage(word_data, x_coord, page_height, gutter_width=min_width)

        if coverage >= 0.80:
            final_separators.append(x_coord)
            print(f"      -> Separator X={x_coord} ACCEPTED (Coverage: {coverage:.1%}, Bridging: {bridging_ratio:.1%})")
        else:
            print(f"      ❌ Separator X={x_coord} REJECTED (Coverage: {coverage:.1%}, Bridging: {bridging_ratio:.1%})")

    return sorted(final_separators)

#======================================================================================================================================


def get_word_data_for_detection(page: fitz.Page, pdf_path: str, page_num: int,
                                top_margin_percent=0.10, bottom_margin_percent=0.10) -> list:
    """
    Retrieves word data using PyMuPDF native extraction or RapidOCR fallback.
    """
    # 1. Attempt Native Extraction
    word_data = page.get_text("words")
    
    if len(word_data) > 5:
        word_data = [(w[4], w[0], w[1], w[2], w[3]) for w in word_data]
    else:
        # 2. Check Cache
        if _ocr_cache.has_ocr(pdf_path, page_num):
            cached_data = _ocr_cache.get_ocr(pdf_path, page_num)
            if cached_data and len(cached_data) > 0:
                return cached_data
        
        # 3. OCR Fallback (RapidOCR)
        try:
            zoom_level = 2.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom_level, zoom_level))
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 3: 
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif pix.n == 4: 
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            
            # CRITICAL FIX: Use return_word_box=True and access word_results
            ocr_result = ocr_engine(img_np, return_word_box=True)
            
            full_word_data = []
            
            # Check if we got valid results
            if ocr_result and ocr_result.word_results:
                scale_adjustment = 1.0 / zoom_level

                
                # Flatten the per-line word results
                flat_results = sum(ocr_result.word_results, ())
                
                for text, score, bbox in flat_results:
                    text = str(text).strip()
                    
                    if text:
                        # Convert Polygon to BBox
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        
                        x1 = min(xs) * scale_adjustment
                        y1 = min(ys) * scale_adjustment
                        x2 = max(xs) * scale_adjustment
                        y2 = max(ys) * scale_adjustment
                        
                        full_word_data.append((text, x1, y1, x2, y2))
            
            word_data = full_word_data
            
            if len(word_data) > 0:
                _ocr_cache.set_ocr(pdf_path, page_num, word_data)
            
        except Exception as e:
            print(f"   ❌ RapidOCR Error in detection phase: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # 4. Apply Margin Filtering
    page_height = page.rect.height
    y_min = page_height * top_margin_percent
    y_max = page_height * (1 - bottom_margin_percent)
    
    return [d for d in word_data if d[2] >= y_min and d[4] <= y_max]

#=========================================================================================================================================
#=============================================================================================================================================








def pixmap_to_numpy(pix: fitz.Pixmap) -> np.ndarray:
    img_data = pix.samples
    img = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img








def extract_native_words_and_convert(fitz_page, scale_factor: float = 2.0) -> list:
    # 1. Get raw data
    try:
        raw_word_data = fitz_page.get_text("words")
    except Exception as e:
        print(f"  ❌ PyMuPDF extraction failed completely: {e}")
        return []

    # ==============================================================================
    # --- DEBUGGING BLOCK: CHECK FIRST 50 NATIVE WORDS (SAFE PRINT) ---
    # ==============================================================================
    print(f"\n[DEBUG] Native Extraction (Page {fitz_page.number + 1}): Checking first 50 words...")
    
    debug_count = 0
    for item in raw_word_data:
        if debug_count >= 50: break
        
        word_text = item[4]
        
        # --- SAFE PRINTING LOGIC ---
        # We encode/decode to ignore surrogates just for the print statement
        # This prevents the "UnicodeEncodeError" that was crashing your script
        safe_text = word_text.encode('utf-8', 'ignore').decode('utf-8')
        
        # Get hex codes (handling potential errors in 'ord')
        try:
            unicode_points = [f"\\u{ord(c):04x}" for c in word_text]
        except:
            unicode_points = ["ERROR"]
            
        print(f"  Word {debug_count}: '{safe_text}' -> Codes: {unicode_points}")
        debug_count += 1
    print("----------------------------------------------------------------------\n")
    # ==============================================================================

    converted_ocr_output = []
    DEFAULT_CONFIDENCE = 99.0

    for x1, y1, x2, y2, word, *rest in raw_word_data:
        # --- FIX: ROBUST SANITIZATION ---
        # 1. Encode to UTF-8 ignoring errors (strips surrogates)
        # 2. Decode back to string
        cleaned_word_bytes = word.encode('utf-8', 'ignore')
        cleaned_word = cleaned_word_bytes.decode('utf-8')
        cleaned_word = word.encode('utf-8', 'ignore').decode('utf-8').strip()
        
        # cleaned_word = cleaned_word.strip()
        if not cleaned_word: continue
        
        x1_pix = int(x1 * scale_factor)
        y1_pix = int(y1 * scale_factor)
        x2_pix = int(x2 * scale_factor)
        y2_pix = int(y2 * scale_factor)
        
        converted_ocr_output.append({
            'type': 'text',
            'word': cleaned_word, 
            'confidence': DEFAULT_CONFIDENCE,
            'bbox': [x1_pix, y1_pix, x2_pix, y2_pix],
            'y0': y1_pix, 'x0': x1_pix
        })
        
    return converted_ocr_output





#===================================================================================================
#===================================================================================================
#===================================================================================================



import pandas as pd
import pickle
import os
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict

# --- Model File Paths (Required for the Classifier to load) ---
VECTORIZER_FILE = 'tfidf_vectorizer_conditional.pkl'
SUBJECT_MODEL_FILE = 'subject_classifier_model_conditional.pkl'
CONDITIONAL_CONCEPT_MODELS_FILE = 'conditional_concept_models.pkl'


# --- Hierarchical Classifier Class (Dependency for the helper function) ---

class HierarchicalClassifier:
    """
    A two-stage classification system based on conditional training.
    Loads the vectorizer, subject classifier, and conditional concept models.
    """

    def __init__(self):
        self.vectorizer = None
        self.subject_model = None
        self.conditional_concept_models = {}
        self.is_ready = False

    def load_models(self):
        """Loads the vectorizer, subject model, and conditional concept models."""
        try:
            start_time = time.time()
            # 1. Load the TF-IDF Vectorizer
            with open(VECTORIZER_FILE, 'rb') as f:
                self.vectorizer = pickle.load(f)

            # 2. Load the Level 1 (Subject) Classifier
            with open(SUBJECT_MODEL_FILE, 'rb') as f:
                self.subject_model = pickle.load(f)

            # 3. Load the dictionary of conditional Level 2 (Concept) Models
            with open(CONDITIONAL_CONCEPT_MODELS_FILE, 'rb') as f:
                conditional_data = pickle.load(f)

                # Extract just the models for easy access
                for subject, data in conditional_data.items():
                    self.conditional_concept_models[subject] = data['model']

            print(f"Loaded models successfully in {time.time() - start_time:.2f} seconds.")
            self.is_ready = True

        except FileNotFoundError as e:
            print(f"Error: Required model file not found: {e.filename}.")
            self.is_ready = False
        except Exception as e:
            print(f"An error occurred while loading models: {e}")
            self.is_ready = False

        return self.is_ready

    def predict_subject(self, text_chunk):
        """Predicts the top Subject (Level 1)."""
        if not self.is_ready:
            return "Unknown", 0.0

        # Vectorize the input
        text_vector = self.vectorizer.transform([text_chunk]).astype(np.float64)

        if hasattr(self.subject_model, 'predict_proba'):
            probabilities = self.subject_model.predict_proba(text_vector)[0]
            classes = self.subject_model.classes_

            top_index = np.argmax(probabilities)
            return classes[top_index], probabilities[top_index]
        else:
            return self.subject_model.predict(text_vector)[0], 1.0

    def predict_concept_hierarchical(self, text_chunk, predicted_subject):
        """
        Predicts the top Concept (Level 2) using the specialized conditional model.
        """
        if not self.is_ready:
            return "Unknown", 0.0

        concept_model = self.conditional_concept_models.get(predicted_subject)

        if concept_model is None or len(getattr(concept_model, 'classes_', [])) <= 1:
            return "No_Conditional_Model_Found", 0.0

        # Vectorize the input
        text_vector = self.vectorizer.transform([text_chunk]).astype(np.float64)

        if hasattr(concept_model, 'predict_proba'):
            probabilities = concept_model.predict_proba(text_vector)[0]
            classes = concept_model.classes_

            top_index = np.argmax(probabilities)
            return classes[top_index], probabilities[top_index]
        else:
            return concept_model.predict(text_vector)[0], 1.0


# --------------------------------------------------------------------------------------
# --- The Requested Helper Function ---

def post_process_json_with_inference(json_data, classifier):
    """
    Takes JSON data, runs hierarchical inference on all question/option text,
    and adds 'predicted_subject' and 'predicted_concept' tags to each entry.

    Args:
        json_data (list): The list of dictionaries containing question entries.
        classifier (HierarchicalClassifier): An initialized and loaded classifier object.

    Returns:
        list: The modified list of dictionaries with classification tags added.
    """
    if not classifier.is_ready:
        print("Classifier not ready. Skipping inference.")
        return json_data

    # This print statement can be removed for silent pipeline integration
    print("\n--- Starting Subject/Concept Detection ---") 

    for entry in json_data:
        # Only process entries that have a 'question' field
        if 'question' not in entry:
            continue

        # 1. Combine Question Text and Option Text for robust feature extraction
        full_text = entry.get('question', '')
        
        options = entry.get('options', {})
        for option_key, option_value in options.items():
            # Use the text component of the option if available
            option_text = option_value if isinstance(option_value, str) else option_key
            full_text += " " + option_text.replace('\n', ' ')

        # Clean up text (remove multiple spaces and surrounding whitespace)
        full_text = ' '.join(full_text.split()).strip()

        # Handle empty text
        if not full_text:
            entry['predicted_subject'] = {'label': 'Empty_Text', 'confidence': 0.0}
            entry['predicted_concept'] = {'label': 'Empty_Text', 'confidence': 0.0}
            continue

        # 2. STAGE 1: Predict Subject
        subj_label, subj_conf = classifier.predict_subject(full_text)
        
        # 3. STAGE 2: Predict Concept (Conditional on predicted subject)
        conc_label, conc_conf = classifier.predict_concept_hierarchical(full_text, subj_label)

        # 4. Add results to the JSON entry
        entry['predicted_subject'] = {
            'label': subj_label,
            'confidence': round(subj_conf, 4)
        }
        entry['predicted_concept'] = {
            'label': conc_label,
            'confidence': round(conc_conf, 4)
        }

    # This print statement can be removed for silent pipeline integration
    # print("--- JSON Post-Processing Complete ---") 
    
    return json_data





#===================================================================================================
#===================================================================================================
#===================================================================================================


    




def preprocess_and_ocr_page(original_img: np.ndarray, model, pdf_path: str,
                            page_num: int, fitz_page: fitz.Page,
                            pdf_name: str) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    OPTIMIZED FLOW:
    1. Run YOLO to find Equations/Tables.
    2. Mask raw text with YOLO boxes.
    3. Run Column Detection on the MASKED data.
    4. Proceed with OCR (Native or High-Res Tesseract Fallback) and Output.
    """
    global GLOBAL_FIGURE_COUNT, GLOBAL_EQUATION_COUNT

    start_time_total = time.time()

    if original_img is None:
        print(f"  ❌ Invalid image for page {page_num}.")
        return None, None

    # ====================================================================
    # --- STEP 1: YOLO DETECTION ---
    # ====================================================================
    start_time_yolo = time.time()
    results = model.predict(source=original_img, conf=0.2, imgsz=640, verbose=False)

    


    relevant_detections = []

    THRESHOLDS = {
        'figure': 0.75,
        'equation': 0.20
    }




    if results and results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])

            # Logic: Check if class is in our list AND meets its specific threshold
            if class_name in THRESHOLDS:
                if conf >= THRESHOLDS[class_name]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    relevant_detections.append({
                        'coords': (x1, y1, x2, y2), 
                        'y1': y1, 
                        'class': class_name, 
                        'conf': conf
                    })

    merged_detections = merge_overlapping_boxes(relevant_detections, IOU_MERGE_THRESHOLD)
    print(f"    [LOG] YOLO found {len(merged_detections)} objects in {time.time() - start_time_yolo:.3f}s.")



    # ====================================================================
    # --- STEP 2: PREPARE DATA FOR COLUMN DETECTION (MASKING) ---
    # ====================================================================
    # Note: This uses the updated 'get_word_data_for_detection' which has its own optimizations
    raw_words_for_layout = get_word_data_for_detection(
        fitz_page, pdf_path, page_num,
        top_margin_percent=0.10, bottom_margin_percent=0.10
    )

    masked_word_data = merge_yolo_into_word_data(raw_words_for_layout, merged_detections, scale_factor=2.0)

    # ====================================================================
    # --- STEP 3: COLUMN DETECTION ---
    # ====================================================================
    page_width_pdf = fitz_page.rect.width
    page_height_pdf = fitz_page.rect.height

    column_detection_params = {
        'cluster_bin_size': 2, 'cluster_smoothing': 2,
        'cluster_min_width': 10, 'cluster_threshold_percentile': 85,
    }

    separators = calculate_x_gutters(masked_word_data, column_detection_params, page_height_pdf)

    page_separator_x = None
    if separators:
        central_min = page_width_pdf * 0.35
        central_max = page_width_pdf * 0.65
        central_separators = [s for s in separators if central_min <= s <= central_max]

        if central_separators:
            center_x = page_width_pdf / 2
            page_separator_x = min(central_separators, key=lambda x: abs(x - center_x))
            print(f"      ✅ Column Split Confirmed at X={page_separator_x:.1f}")
        else:
            print("      ⚠️ Gutter found off-center. Ignoring.")
    else:
        print("      -> Single Column Layout Confirmed.")

    # ====================================================================
    # --- STEP 4: COMPONENT EXTRACTION (Save Images) ---
    # ====================================================================
    start_time_components = time.time()
    component_metadata = []
    fig_count_page = 0
    eq_count_page = 0

    for detection in merged_detections:
        x1, y1, x2, y2 = detection['coords']
        class_name = detection['class']

        if class_name == 'figure':
            GLOBAL_FIGURE_COUNT += 1
            counter = GLOBAL_FIGURE_COUNT
            component_word = f"FIGURE{counter}"
            fig_count_page += 1
        elif class_name == 'equation':
            GLOBAL_EQUATION_COUNT += 1
            counter = GLOBAL_EQUATION_COUNT
            component_word = f"EQUATION{counter}"
            eq_count_page += 1
        else:
            continue

        component_crop = original_img[y1:y2, x1:x2]
        component_filename = f"{pdf_name}_page{page_num}_{class_name}{counter}.png"
        cv2.imwrite(os.path.join(FIGURE_EXTRACTION_DIR, component_filename), component_crop)

        y_midpoint = (y1 + y2) // 2
        component_metadata.append({
            'type': class_name, 'word': component_word,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'y0': int(y_midpoint), 'x0': int(x1)
        })

    # ====================================================================
    # --- STEP 5: HYBRID OCR (Native Text + Cached Tesseract Fallback) ---
    # ====================================================================
    raw_ocr_output = []
    scale_factor = 2.0  # Pipeline standard scale

    try:
        # Try getting native text first
        # NOTE: extract_native_words_and_convert MUST ALSO BE UPDATED TO USE sanitize_text
        raw_ocr_output = extract_native_words_and_convert(fitz_page, scale_factor=scale_factor)
    except Exception as e:
        print(f"  ❌ Native text extraction failed: {e}")

    # If native text is missing, fall back to OCR
    if not raw_ocr_output:
        if _ocr_cache.has_ocr(pdf_path, page_num):
            print(f"  ⚡ Using cached Tesseract OCR for page {page_num}")
            cached_word_data = _ocr_cache.get_ocr(pdf_path, page_num)
            for word_tuple in cached_word_data:
                word_text, x1, y1, x2, y2 = word_tuple

                # Scale from PDF points to Pipeline Pixels (2.0)
                x1_pix = int(x1 * scale_factor)
                y1_pix = int(y1 * scale_factor)
                x2_pix = int(x2 * scale_factor)
                y2_pix = int(y2 * scale_factor)

                raw_ocr_output.append({
                    'type': 'text', 'word': word_text, 'confidence': 95.0,
                    'bbox': [x1_pix, y1_pix, x2_pix, y2_pix],
                    'y0': y1_pix, 'x0': x1_pix
                })
        else:
            # === START OF OPTIMIZED OCR BLOCK ===

#=============================================================================================================================================================
#=============================================================================================================================================================
            
            try:
                # 1. Re-render Page at High Resolution (Standardizing to Zoom 4.0)
                ocr_zoom = 4.0
                pix_ocr = fitz_page.get_pixmap(matrix=fitz.Matrix(ocr_zoom, ocr_zoom))

                # Convert PyMuPDF Pixmap to OpenCV format (BGR)
                img_ocr_np = np.frombuffer(pix_ocr.samples, dtype=np.uint8).reshape(
                    pix_ocr.height, pix_ocr.width, pix_ocr.n
                )
                if pix_ocr.n == 3:
                    img_ocr_np = cv2.cvtColor(img_ocr_np, cv2.COLOR_RGB2BGR)
                elif pix_ocr.n == 4:
                    img_ocr_np = cv2.cvtColor(img_ocr_np, cv2.COLOR_RGBA2BGR)

                # 2. Run RapidOCR
                # FIX 1: Capture the object (Removes the "cannot unpack" error)
                ocr_out = ocr_engine(img_ocr_np)

                # FIX 2: Use 'is not None' (Removes the "ambiguous truth value" error)
                if ocr_out is not None and ocr_out.boxes is not None:
                    # Calculate scaling from OCR image (4.0) to your pipeline standard (scale_factor=2.0)
                    scale_adjustment = scale_factor / ocr_zoom

                    # FIX 3: Zip the attributes to restore your expected (box, text, score) format
                    for box, text, score in zip(ocr_out.boxes, ocr_out.txts, ocr_out.scores):
                        # Sanitize and clean text
                        cleaned_text = sanitize_text(str(text)).strip()
                        
                        if cleaned_text:
                            # 3. Coordinate Mapping (Convert 4-point polygon to x1, y1, x2, y2)
                            xs = [p[0] for p in box]
                            ys = [p[1] for p in box]
                            
                            x1 = int(min(xs) * scale_adjustment)
                            y1 = int(min(ys) * scale_adjustment)
                            x2 = int(max(xs) * scale_adjustment)
                            y2 = int(max(ys) * scale_adjustment)

                            raw_ocr_output.append({
                                'type': 'text',
                                'word': cleaned_text,
                                'confidence': float(score) * 100, # Converting 0-1.0 to 0-100 scale
                                'bbox': [x1, y1, x2, y2],
                                'y0': y1,
                                'x0': x1
                            })
            except Exception as e:
                print(f"  ❌ RapidOCR Fallback Error: {e}")
            # === END OF RAPIDOCR BLOCK ==========================
            # === END OF RAPIDOCR BLOCK ====================================================================================================================================
#===========================================================================================================================================================================
            # === END OF OPTIMIZED OCR BLOCK ===
    
    # ====================================================================
    # --- STEP 6: OCR CLEANING AND MERGING ---
    # ====================================================================
    items_to_sort = []

    for ocr_word in raw_ocr_output:
        is_suppressed = False
        for component in component_metadata:
            # Do not include words that are inside figure/equation boxes
            ioa = calculate_ioa(ocr_word['bbox'], component['bbox'])
            if ioa > IOA_SUPPRESSION_THRESHOLD:
                is_suppressed = True
                break
        if not is_suppressed:
            items_to_sort.append(ocr_word)

    # Add figures/equations back into the flow as "words"
    items_to_sort.extend(component_metadata)

    # ====================================================================
    # --- STEP 7: LINE-BASED SORTING ---
    # ====================================================================
    items_to_sort.sort(key=lambda x: (x['y0'], x['x0']))
    lines = []

    for item in items_to_sort:
        placed = False
        for line in lines:
            y_ref = min(it['y0'] for it in line)
            if abs(y_ref - item['y0']) < LINE_TOLERANCE:
                line.append(item)
                placed = True
                break
        if not placed and item['type'] in ['equation', 'figure']:
            for line in lines:
                y_ref = min(it['y0'] for it in line)
                if abs(y_ref - item['y0']) < 20:
                    line.append(item)
                    placed = True
                    break
        if not placed:
            lines.append([item])

    for line in lines:
        line.sort(key=lambda x: x['x0'])

    final_output = []
    for line in lines:
        for item in line:
            data_item = {"word": item["word"], "bbox": item["bbox"], "type": item["type"]}
            if 'tag' in item: data_item['tag'] = item['tag']
            final_output.append(data_item)

    return final_output, page_separator_x








def run_single_pdf_preprocessing(pdf_path: str, preprocessed_json_path: str) -> Optional[str]:
    global GLOBAL_FIGURE_COUNT, GLOBAL_EQUATION_COUNT

    GLOBAL_FIGURE_COUNT = 0
    GLOBAL_EQUATION_COUNT = 0
    _ocr_cache.clear()

    print("\n" + "=" * 80)
    print("--- 1. STARTING OPTIMIZED YOLO/OCR PREPROCESSING PIPELINE ---")
    print("=" * 80)

    if not os.path.exists(pdf_path):
        print(f"❌ FATAL ERROR: Input PDF not found at {pdf_path}.")
        return None

    os.makedirs(os.path.dirname(preprocessed_json_path), exist_ok=True)
    os.makedirs(FIGURE_EXTRACTION_DIR, exist_ok=True)

    model = YOLO(WEIGHTS_PATH)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        doc = fitz.open(pdf_path)
        print(f"✅ Opened PDF: {pdf_name} ({doc.page_count} pages)")
    except Exception as e:
        print(f"❌ ERROR loading PDF file: {e}")
        return None

    all_pages_data = []
    total_pages_processed = 0
    mat = fitz.Matrix(2.0, 2.0)

    print("\n[STEP 1.2: ITERATING PAGES - IN-MEMORY PROCESSING]")

    for page_num_0_based in range(doc.page_count):
        page_num = page_num_0_based + 1
        print(f"  -> Processing Page {page_num}/{doc.page_count}...")

        fitz_page = doc.load_page(page_num_0_based)

        try:
            pix = fitz_page.get_pixmap(matrix=mat)
            original_img = pixmap_to_numpy(pix)
        except Exception as e:
            print(f"  ❌ Error converting page {page_num} to image: {e}")
            continue

        final_output, page_separator_x = preprocess_and_ocr_page(
            original_img,
            model,
            pdf_path,
            page_num,
            fitz_page,
            pdf_name
        )

        if final_output is not None:
            page_data = {
                "page_number": page_num,
                "data": final_output,
                "column_separator_x": page_separator_x
            }
            all_pages_data.append(page_data)
            total_pages_processed += 1
        else:
            print(f"  ❌ Skipped page {page_num} due to processing error.")

    doc.close()

    if all_pages_data:
        try:
            with open(preprocessed_json_path, 'w') as f:
                json.dump(all_pages_data, f, indent=4)
            print(f"\n  ✅ Combined structured OCR JSON saved to: {os.path.basename(preprocessed_json_path)}")
        except Exception as e:
            print(f"❌ ERROR saving combined JSON output: {e}")
            return None
    else:
        print("❌ WARNING: No page data generated. Halting pipeline.")
        return None

    print("\n" + "=" * 80)
    print(f"--- YOLO/OCR PREPROCESSING COMPLETE ({total_pages_processed} pages processed) ---")
    print("=" * 80)

    return preprocessed_json_path


# ============================================================================
# --- PHASE 2: LAYOUTLMV3 INFERENCE FUNCTIONS ---
# ============================================================================

class LayoutLMv3ForTokenClassification(nn.Module):
    def __init__(self, num_labels: int = NUM_LABELS):
        super().__init__()
        self.num_labels = num_labels
        config = LayoutLMv3Config.from_pretrained("microsoft/layoutlmv3-base", num_labels=num_labels)
        self.layoutlmv3 = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base", config=config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None: nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, bbox: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        outputs = self.layoutlmv3(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state
        emissions = self.classifier(sequence_output)
        mask = attention_mask.bool()
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask).mean()
            return loss
        else:
            return self.crf.viterbi_decode(emissions, mask=mask)


def _merge_integrity(all_token_data: List[Dict[str, Any]],
                     column_separator_x: Optional[int]) -> List[List[Dict[str, Any]]]:
    """Splits the token data objects into column chunks based on a separator."""
    if column_separator_x is None:
        print("    -> No column separator. Treating as one chunk.")
        return [all_token_data]

    left_column_tokens, right_column_tokens = [], []
    for token_data in all_token_data:
        bbox_raw = token_data['bbox_raw_pdf_space']
        center_x = (bbox_raw[0] + bbox_raw[2]) / 2
        if center_x < column_separator_x:
            left_column_tokens.append(token_data)
        else:
            right_column_tokens.append(token_data)

    chunks = [c for c in [left_column_tokens, right_column_tokens] if c]
    print(f"    -> Data split into {len(chunks)} column chunk(s) using separator X={column_separator_x}.")
    return chunks






# def run_inference_and_get_raw_words(pdf_path: str, model_path: str,
#                                     preprocessed_json_path: str,
#                                     column_detection_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
#     print("\n" + "=" * 80)
#     print("--- 2. STARTING LAYOUTLMV3 INFERENCE PIPELINE (Raw Word Output) ---")
#     print("=" * 80)

#     tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"  -> Using device: {device}")

#     try:
#         model = LayoutLMv3ForTokenClassification(num_labels=NUM_LABELS)
#         checkpoint = torch.load(model_path, map_location=device)
#         model_state = checkpoint.get('model_state_dict', checkpoint)
#         # Apply patch for layoutlmv3 compatibility with saved state_dict
#         fixed_state_dict = {key.replace('layoutlm.', 'layoutlmv3.'): value for key, value in model_state.items()}
#         model.load_state_dict(fixed_state_dict)
#         model.to(device)
#         model.eval()
#         print(f"✅ LayoutLMv3 Model loaded successfully from {os.path.basename(model_path)}.")
#     except Exception as e:
#         print(f"❌ FATAL ERROR during LayoutLMv3 model loading: {e}")
#         return []

#     try:
#         with open(preprocessed_json_path, 'r', encoding='utf-8') as f:
#             preprocessed_data = json.load(f)
#         print(f"✅ Loaded preprocessed data with {len(preprocessed_data)} pages.")
#     except Exception:
#         print("❌ Error loading preprocessed JSON.")
#         return []

#     try:
#         doc = fitz.open(pdf_path)
#     except Exception:
#         print("❌ Error loading PDF.")
#         return []

#     final_page_predictions = []
#     CHUNK_SIZE = 500

#     for page_data in preprocessed_data:
#         page_num_1_based = page_data['page_number']
#         page_num_0_based = page_num_1_based - 1
#         page_raw_predictions = []
#         print(f"\n  *** Processing Page {page_num_1_based} ({len(page_data['data'])} raw tokens) ***")

#         fitz_page = doc.load_page(page_num_0_based)
#         page_width, page_height = fitz_page.rect.width, fitz_page.rect.height
#         print(f"    -> Page dimensions: {page_width:.0f}x{page_height:.0f} (PDF points).")

#         all_token_data = []
#         scale_factor = 2.0

#         for item in page_data['data']:
#             raw_yolo_bbox = item['bbox']
#             bbox_pdf = [
#                 int(raw_yolo_bbox[0] / scale_factor), int(raw_yolo_bbox[1] / scale_factor),
#                 int(raw_yolo_bbox[2] / scale_factor), int(raw_yolo_bbox[3] / scale_factor)
#             ]
#             normalized_bbox = [
#                 max(0, min(1000, int(1000 * bbox_pdf[0] / page_width))),
#                 max(0, min(1000, int(1000 * bbox_pdf[1] / page_height))),
#                 max(0, min(1000, int(1000 * bbox_pdf[2] / page_width))),
#                 max(0, min(1000, int(1000 * bbox_pdf[3] / page_height)))
#             ]
#             all_token_data.append({
#                 "word": item['word'],
#                 "bbox_raw_pdf_space": bbox_pdf,
#                 "bbox_normalized": normalized_bbox,
#                 "item_original_data": item
#             })

#         if not all_token_data:
#             continue

#         column_separator_x = page_data.get('column_separator_x', None)
#         if column_separator_x is not None:
#             print(f"    -> Using SAVED column separator: X={column_separator_x}")
#         else:
#             print("    -> No column separator found. Assuming single chunk.")

#         token_chunks = _merge_integrity(all_token_data, column_separator_x)
#         total_chunks = len(token_chunks)

#         for chunk_idx, chunk_tokens in enumerate(token_chunks):
#             if not chunk_tokens: continue

#             # 1. Sanitize: Convert everything to strings and aggressively clean Unicode errors.
#             chunk_words = [
#                 str(t['word']).encode('utf-8', errors='ignore').decode('utf-8')
#                 for t in chunk_tokens
#             ]
#             chunk_normalized_bboxes = [t['bbox_normalized'] for t in chunk_tokens]

#             total_sub_chunks = (len(chunk_words) + CHUNK_SIZE - 1) // CHUNK_SIZE
#             for i in range(0, len(chunk_words), CHUNK_SIZE):
#                 sub_chunk_idx = i // CHUNK_SIZE + 1
#                 sub_words = chunk_words[i:i + CHUNK_SIZE]
#                 sub_bboxes = chunk_normalized_bboxes[i:i + CHUNK_SIZE]
#                 sub_tokens_data = chunk_tokens[i:i + CHUNK_SIZE]

#                 print(f"      -> Chunk {chunk_idx + 1}/{total_chunks}, Sub-chunk {sub_chunk_idx}/{total_sub_chunks}: {len(sub_words)} words. Running Inference...")

#                 # 2. Manual generation of word_ids
#                 manual_word_ids = []
#                 for current_word_idx, word in enumerate(sub_words):
#                     sub_tokens = tokenizer.tokenize(word)
#                     for _ in sub_tokens:
#                         manual_word_ids.append(current_word_idx)

#                 encoded_input = tokenizer(
#                     sub_words,
#                     boxes=sub_bboxes,
#                     truncation=True,
#                     padding="max_length",
#                     max_length=512,
#                     is_split_into_words=True,
#                     return_tensors="pt"
#                 )

#                 # Check for empty sequence
#                 if encoded_input['input_ids'].shape[0] == 0:
#                     print(f"        -> Warning: Sub-chunk {sub_chunk_idx} encoded to an empty sequence. Skipping.")
#                     continue

#                 # 3. Finalize word_ids based on encoded output length
#                 sequence_length = int(torch.sum(encoded_input['attention_mask']).item())
#                 content_token_length = max(0, sequence_length - 2)

#                 manual_word_ids = manual_word_ids[:content_token_length]

#                 final_word_ids = [None]  # CLS token (index 0)
#                 final_word_ids.extend(manual_word_ids)

#                 if sequence_length > 1:
#                     final_word_ids.append(None)  # SEP token

#                 final_word_ids.extend([None] * (512 - len(final_word_ids)))
#                 word_ids = final_word_ids[:512]  # Final array for mapping

#                 # Inputs are already batched by the tokenizer as [1, 512] 
#                 input_ids = encoded_input['input_ids'].to(device)
#                 bbox = encoded_input['bbox'].to(device)
#                 attention_mask = encoded_input['attention_mask'].to(device)

#                 with torch.no_grad():
#                     model_outputs = model(input_ids, bbox, attention_mask)

#                 # --- Robust extraction: support several forward return types ---
#                 # We'll try (in order):
#                 # 1) model_outputs is (emissions_tensor, viterbi_list)  -> use emissions for logits, keep decoded
#                 # 2) model_outputs has .logits attribute (HF ModelOutput)
#                 # 3) model_outputs is tuple/list containing a logits tensor
#                 # 4) model_outputs is a tensor (assume logits)
#                 # 5) model_outputs is a list-of-lists of ints (viterbi decoded) -> use that directly (no logits)
#                 logits_tensor = None
#                 decoded_labels_list = None

#                 # case 1: tuple/list with (emissions, viterbi)
#                 if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 2:
#                     a, b = model_outputs
#                     # a might be tensor (emissions), b might be viterbi list
#                     if isinstance(a, torch.Tensor):
#                         logits_tensor = a
#                     if isinstance(b, list):
#                         decoded_labels_list = b

#                 # case 2: HF ModelOutput with .logits
#                 if logits_tensor is None and hasattr(model_outputs, 'logits') and isinstance(model_outputs.logits, torch.Tensor):
#                     logits_tensor = model_outputs.logits

#                 # case 3: tuple/list - search for a 3D tensor (B, L, C)
#                 if logits_tensor is None and isinstance(model_outputs, (tuple, list)):
#                     found_tensor = None
#                     for item in model_outputs:
#                         if isinstance(item, torch.Tensor):
#                             # prefer 3D (batch, seq, labels)
#                             if item.dim() == 3:
#                                 logits_tensor = item
#                                 break
#                             if found_tensor is None:
#                                 found_tensor = item
#                     if logits_tensor is None and found_tensor is not None:
#                         # found_tensor may be (batch, seq, hidden) or (seq, hidden); we avoid guessing.
#                         # Keep found_tensor only if it matches num_labels dimension
#                         if found_tensor.dim() == 3 and found_tensor.shape[-1] == NUM_LABELS:
#                             logits_tensor = found_tensor
#                         elif found_tensor.dim() == 2 and found_tensor.shape[-1] == NUM_LABELS:
#                             logits_tensor = found_tensor.unsqueeze(0)

#                 # case 4: model_outputs directly a tensor
#                 if logits_tensor is None and isinstance(model_outputs, torch.Tensor):
#                     logits_tensor = model_outputs

#                 # case 5: model_outputs is a decoded viterbi list (common for CRF-only forward)
#                 if decoded_labels_list is None and isinstance(model_outputs, list) and model_outputs and isinstance(model_outputs[0], list):
#                     # assume model_outputs is already viterbi decoded: List[List[int]] with batch dim first
#                     decoded_labels_list = model_outputs

#                 # If neither logits nor decoded exist, that's fatal
#                 if logits_tensor is None and decoded_labels_list is None:
#                     # helpful debug info
#                     try:
#                         elem_shapes = [ (type(x), getattr(x, 'shape', None)) for x in model_outputs ] if isinstance(model_outputs, (list, tuple)) else [(type(model_outputs), getattr(model_outputs, 'shape', None))]
#                     except Exception:
#                         elem_shapes = str(type(model_outputs))
#                     raise RuntimeError(f"Model output of type {type(model_outputs)} did not contain a valid logits tensor or decoded viterbi. Contents: {elem_shapes}")

#                 # If we have logits_tensor, normalize shape to [seq_len, num_labels]
#                 if logits_tensor is not None:
#                     # If shape is [B, L, C] with B==1, squeeze batch
#                     if logits_tensor.dim() == 3 and logits_tensor.shape[0] == 1:
#                         preds_tensor = logits_tensor.squeeze(0)  # [L, C]
#                     else:
#                         preds_tensor = logits_tensor  # possibly [L, C] already

#                     # Safety: ensure we have at least seq_len x channels
#                     if preds_tensor.dim() != 2:
#                         # try to reshape or error
#                         raise RuntimeError(f"Unexpected logits tensor shape: {tuple(preds_tensor.shape)}")
#                     # We'll use preds_tensor[token_idx] to argmax
#                 else:
#                     preds_tensor = None  # no logits available

#                 # If decoded labels provided, make a token-level list-of-ints aligned to tokenizer tokens
#                 decoded_token_labels = None
#                 if decoded_labels_list is not None:
#                     # decoded_labels_list is batch-first; we used batch size 1
#                     # if multiple sequences returned, take first
#                     decoded_token_labels = decoded_labels_list[0] if isinstance(decoded_labels_list[0], list) else decoded_labels_list

#                 # Now map token-level predictions -> word-level predictions using word_ids
#                 word_idx_to_pred_id = {}

#                 if preds_tensor is not None:
#                     # We have logits. Use argmax of logits for each token id up to sequence_length
#                     for token_idx, word_idx in enumerate(word_ids):
#                         if token_idx >= sequence_length:
#                             break
#                         if word_idx is not None and word_idx < len(sub_words):
#                             if word_idx not in word_idx_to_pred_id:
#                                 pred_id = torch.argmax(preds_tensor[token_idx]).item()
#                                 word_idx_to_pred_id[word_idx] = pred_id
#                 else:
#                     # No logits, but we have decoded_token_labels from CRF (one label per token)
#                     # We'll align decoded_token_labels to token positions.
#                     if decoded_token_labels is None:
#                         # should not happen due to earlier checks
#                         raise RuntimeError("No logits and no decoded labels available for mapping.")
#                     # decoded_token_labels length may be equal to content_token_length (no special tokens)
#                     # or equal to sequence_length; try to align intelligently:
#                     # Prefer using decoded_token_labels aligned to the tokenizer tokens (starting at token 1 for CLS)
#                     # If decoded length == content_token_length, then manual_word_ids maps sub-token -> word idx for content tokens only.
#                     # We'll iterate tokens and pick label accordingly.
#                     # Build token_idx -> decoded_label mapping:
#                     # We'll assume decoded_token_labels correspond to content tokens (no CLS/SEP). If decoded length == sequence_length, then shift by 0.
#                     decoded_len = len(decoded_token_labels)
#                     # Heuristic: if decoded_len == content_token_length -> alignment starts at token_idx 1 (skip CLS)
#                     if decoded_len == content_token_length:
#                         decoded_start = 1
#                     elif decoded_len == sequence_length:
#                         decoded_start = 0
#                     else:
#                         # fallback: prefer decoded_start=1 (most common)
#                         decoded_start = 1

#                     for tok_idx_in_decoded, label_id in enumerate(decoded_token_labels):
#                         tok_idx = decoded_start + tok_idx_in_decoded
#                         if tok_idx >= 512:
#                             break
#                         if tok_idx >= sequence_length:
#                             break
#                         # map this token to a word index if present
#                         word_idx = word_ids[tok_idx] if tok_idx < len(word_ids) else None
#                         if word_idx is not None and word_idx < len(sub_words):
#                             if word_idx not in word_idx_to_pred_id:
#                                 # label_id may already be an int
#                                 word_idx_to_pred_id[word_idx] = int(label_id)

#                 # Finally convert mapped word preds -> page_raw_predictions entries
#                 for current_word_idx in range(len(sub_words)):
#                     pred_id = word_idx_to_pred_id.get(current_word_idx, 0)  # default to 0
#                     predicted_label = ID_TO_LABEL[pred_id]
#                     original_token = sub_tokens_data[current_word_idx]
#                     page_raw_predictions.append({
#                         "word": original_token['word'],
#                         "bbox": original_token['bbox_raw_pdf_space'],
#                         "predicted_label": predicted_label,
#                         "page_number": page_num_1_based
#                     })

#         if page_raw_predictions:
#             final_page_predictions.append({
#                 "page_number": page_num_1_based,
#                 "data": page_raw_predictions
#             })
#             print(f"  *** Page {page_num_1_based} Finalized: {len(page_raw_predictions)} labeled words. ***")

#     doc.close()
#     print("\n" + "=" * 80)
#     print("--- LAYOUTLMV3 INFERENCE COMPLETE ---")
#     print("=" * 80)
#     return final_page_predictions








def run_inference_and_get_raw_words(pdf_path: str, model_path: str,
                                    preprocessed_json_path: str,
                                    column_detection_params: Optional[Dict] = None) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("--- 2. STARTING LAYOUTLMV3 INFERENCE PIPELINE (Sliding Window) ---")
    print("=" * 80)

    # 1. Setup
    tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  -> Using device: {device}")

    # 2. Load Model
    try:
        model = LayoutLMv3ForTokenClassification(num_labels=NUM_LABELS)
        checkpoint = torch.load(model_path, map_location=device)
        model_state = checkpoint.get('model_state_dict', checkpoint)
        fixed_state_dict = {key.replace('layoutlm.', 'layoutlmv3.'): value for key, value in model_state.items()}
        model.load_state_dict(fixed_state_dict)
        model.to(device)
        model.eval()
        print(f"✅ LayoutLMv3 Model loaded successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR during LayoutLMv3 model loading: {e}")
        return []

    # 3. Load Data
    try:
        with open(preprocessed_json_path, 'r', encoding='utf-8') as f:
            preprocessed_data = json.load(f)
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        return []

    final_page_predictions = []

    # =========================================================================
    # ⚙️ SLIDING WINDOW SETTINGS
    # We use a window of 350 words. Since 1 word ≈ 1.3 tokens, 
    # 350 words ≈ 455 tokens. This leaves a ~50 token buffer for safety 
    # against the 512 limit, preventing "trailing off".
    # =========================================================================
    WINDOW_SIZE_WORDS = 350  
    STRIDE_WORDS = 250       # Overlap of 100 words (350 - 250)

    for page_data in preprocessed_data:
        page_num_1_based = page_data['page_number']
        page_num_0_based = page_num_1_based - 1
        page_final_words = []
        
        print(f"\n  *** Processing Page {page_num_1_based} ({len(page_data['data'])} raw tokens) ***")

        fitz_page = doc.load_page(page_num_0_based)
        page_width, page_height = fitz_page.rect.width, fitz_page.rect.height
        
        # --- A. PREPARE ALL DATA FOR PAGE ---
        all_token_data = []
        scale_factor = 2.0

        for item in page_data['data']:
            raw_yolo_bbox = item['bbox']
            bbox_pdf = [
                int(raw_yolo_bbox[0] / scale_factor), int(raw_yolo_bbox[1] / scale_factor),
                int(raw_yolo_bbox[2] / scale_factor), int(raw_yolo_bbox[3] / scale_factor)
            ]
            normalized_bbox = [
                max(0, min(1000, int(1000 * bbox_pdf[0] / page_width))),
                max(0, min(1000, int(1000 * bbox_pdf[1] / page_height))),
                max(0, min(1000, int(1000 * bbox_pdf[2] / page_width))),
                max(0, min(1000, int(1000 * bbox_pdf[3] / page_height)))
            ]
            all_token_data.append({
                "word": str(item['word']).encode('utf-8', errors='ignore').decode('utf-8'),
                "bbox_raw_pdf_space": bbox_pdf,
                "bbox_normalized": normalized_bbox,
                "item_original_data": item
            })

        if not all_token_data:
            continue

        column_separator_x = page_data.get('column_separator_x', None)
        token_chunks = _merge_integrity(all_token_data, column_separator_x)
        total_chunks = len(token_chunks)

        # --- B. PROCESS EACH COLUMN CHUNK WITH SLIDING WINDOW ---
        for chunk_idx, chunk_tokens in enumerate(token_chunks):
            if not chunk_tokens: continue
            
            # Use a dict to store predictions: {word_index_in_chunk: label}
            # This handles the stitching automatically.
            chunk_predictions_map = {}

            chunk_words = [t['word'] for t in chunk_tokens]
            chunk_bboxes = [t['bbox_normalized'] for t in chunk_tokens]
            num_words_in_chunk = len(chunk_words)

            print(f"      -> Chunk {chunk_idx + 1}/{total_chunks}: {num_words_in_chunk} words.")

            # Loop with stride (0, 250, 500, 750...)
            for start_idx in range(0, num_words_in_chunk, STRIDE_WORDS):
                end_idx = min(start_idx + WINDOW_SIZE_WORDS, num_words_in_chunk)
                
                # Check if this window is entirely redundant (optimization)
                if start_idx > 0 and end_idx == num_words_in_chunk and (end_idx - start_idx) < (WINDOW_SIZE_WORDS - STRIDE_WORDS):
                     pass # You can opt to skip very small tails if fully covered, but safer to run.

                sub_words = chunk_words[start_idx:end_idx]
                sub_bboxes = chunk_bboxes[start_idx:end_idx]

                # --- 1. Tokenize & Encode ---
                manual_word_ids = []
                for local_word_idx, word in enumerate(sub_words):
                    sub_tokens = tokenizer.tokenize(word)
                    for _ in sub_tokens:
                        manual_word_ids.append(local_word_idx)

                encoded_input = tokenizer(
                    sub_words,
                    boxes=sub_bboxes,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    is_split_into_words=True,
                    return_tensors="pt"
                )

                if encoded_input['input_ids'].shape[0] == 0:
                    continue

                # --- 2. Align Tokens ---
                sequence_length = int(torch.sum(encoded_input['attention_mask']).item())
                content_token_length = max(0, sequence_length - 2) 
                
                manual_word_ids = manual_word_ids[:content_token_length]

                final_word_ids = [None] # CLS
                final_word_ids.extend(manual_word_ids)
                if sequence_length > 1: final_word_ids.append(None) # SEP
                final_word_ids.extend([None] * (512 - len(final_word_ids)))
                word_ids = final_word_ids[:512]

                # --- 3. Inference ---
                input_ids = encoded_input['input_ids'].to(device)
                bbox = encoded_input['bbox'].to(device)
                attention_mask = encoded_input['attention_mask'].to(device)

                with torch.no_grad():
                    model_outputs = model(input_ids, bbox, attention_mask)

                # --- 4. Unpack Outputs (Robust) ---
                logits_tensor = None
                decoded_labels_list = None

                # Handle various return types from CRF/Model
                if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 2:
                    a, b = model_outputs
                    if isinstance(a, torch.Tensor): logits_tensor = a
                    if isinstance(b, list): decoded_labels_list = b
                if logits_tensor is None and hasattr(model_outputs, 'logits'):
                    logits_tensor = model_outputs.logits
                if logits_tensor is None and isinstance(model_outputs, torch.Tensor):
                    logits_tensor = model_outputs
                if decoded_labels_list is None and isinstance(model_outputs, list) and model_outputs and isinstance(model_outputs[0], list):
                    decoded_labels_list = model_outputs

                # --- 5. Map Predictions to Words ---
                local_window_preds = {} # {local_word_idx: pred_id}

                if preds_tensor := logits_tensor:
                    # Logits Path
                    if preds_tensor.dim() == 3 and preds_tensor.shape[0] == 1: preds_tensor = preds_tensor.squeeze(0)
                    for token_idx, word_idx in enumerate(word_ids):
                        if token_idx >= sequence_length: break
                        if word_idx is not None and word_idx < len(sub_words):
                            if word_idx not in local_window_preds:
                                pred_id = torch.argmax(preds_tensor[token_idx]).item()
                                local_window_preds[word_idx] = pred_id
                elif decoded_token_labels := (decoded_labels_list[0] if decoded_labels_list else None):
                    # Viterbi Path
                    decoded_start = 1 if len(decoded_token_labels) == content_token_length else 0
                    for tok_idx_in_decoded, label_id in enumerate(decoded_token_labels):
                        tok_idx = decoded_start + tok_idx_in_decoded
                        if tok_idx >= 512: break
                        word_idx = word_ids[tok_idx] if tok_idx < len(word_ids) else None
                        if word_idx is not None and word_idx < len(sub_words):
                            if word_idx not in local_window_preds:
                                local_window_preds[word_idx] = int(label_id)

                # --- 6. Aggregate into Global Map (Stitching) ---
                for local_idx, pred_id in local_window_preds.items():
                    global_chunk_idx = start_idx + local_idx
                    if global_chunk_idx < num_words_in_chunk:
                        # Overwrite previous predictions. The "later" window usually has better
                        # left-context for these words than the previous window had right-context.
                        chunk_predictions_map[global_chunk_idx] = ID_TO_LABEL[pred_id]

            # --- C. FINALIZE CHUNK ---
            for i in range(num_words_in_chunk):
                predicted_label = chunk_predictions_map.get(i, "O")
                original_token = chunk_tokens[i]
                
                page_final_words.append({
                    "word": original_token['word'],
                    "bbox": original_token['bbox_raw_pdf_space'],
                    "predicted_label": predicted_label,
                    "page_number": page_num_1_based
                })

        if page_final_words:
            final_page_predictions.append({
                "page_number": page_num_1_based,
                "data": page_final_words
            })
            print(f"  *** Page {page_num_1_based} Finalized: {len(page_final_words)} labeled words. ***")

    doc.close()
    print("\n" + "=" * 80)
    print("--- LAYOUTLMV3 INFERENCE COMPLETE ---")
    print("=" * 80)
    return final_page_predictions







# ============================================================================
# --- PHASE 2 REPLACEMENT: CUSTOM INFERENCE PIPELINE ---
# ============================================================================
def run_custom_inference_and_get_raw_words(preprocessed_json_path: str) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("--- 2. STARTING CUSTOM MODEL INFERENCE PIPELINE ---")
    print("=" * 80)

    # 1. Load Resources
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VOCAB_FILE):
        print("❌ Error: Missing custom model or vocab files.")
        return []

    try:
        print("  -> Loading Vocab and Model...")
        with open(VOCAB_FILE, "rb") as f:
            word_vocab, char_vocab = pickle.load(f)
        
        model = MCQTagger(len(word_vocab), len(char_vocab), len(LABELS)).to(DEVICE)
        
        # Load state dict safe
        state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
        model.load_state_dict(state_dict if isinstance(state_dict, dict) else state_dict.state_dict())
        model.eval()
        print("✅ Custom Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading custom model: {e}")
        return []

    # 2. Load Preprocessed Data
    try:
        with open(preprocessed_json_path, 'r', encoding='utf-8') as f:
            preprocessed_data = json.load(f)
        print(f"✅ Loaded preprocessed data for {len(preprocessed_data)} pages.")
    except Exception:
        print("❌ Error loading preprocessed JSON.")
        return []

    final_page_predictions = []
    scale_factor = 2.0  # The pipeline scales PDF points to 2.0 for YOLO. We need to reverse this.

    for page_data in preprocessed_data:
        page_num = page_data['page_number']
        raw_items = page_data['data']
        
        if not raw_items: continue

        # --- A. ADAPTER: Convert Pipeline Data format to Custom Model format ---
        # Pipeline Data: {'word': 'Text', 'bbox': [x1, y1, x2, y2]} (scaled by 2.0)
        # Custom Model Needed: {'word': 'Text', 'x0': x, 'y0': y, 'x1': x, 'y1': y} (PDF points)
        
        tokens_for_inference = []
        for item in raw_items:
            bbox = item['bbox']
            # Revert scale to get native PDF coordinates
            x0 = bbox[0] / scale_factor
            y0 = bbox[1] / scale_factor
            x1 = bbox[2] / scale_factor
            y1 = bbox[3] / scale_factor
            
            tokens_for_inference.append({
                'word': str(item['word']), # Ensure string
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'original_bbox': bbox # Keep for output
            })

        # --- B. FEATURE EXTRACTION ---
        for i in range(len(tokens_for_inference)):
            tokens_for_inference[i]['spatial_features'] = extract_spatial_features(tokens_for_inference, i)
            tokens_for_inference[i]['context_features'] = extract_context_features(tokens_for_inference, i)

        # --- C. BATCH INFERENCE ---
        page_raw_predictions = []
        
        # Process in chunks
        for i in range(0, len(tokens_for_inference), INFERENCE_CHUNK_SIZE):
            chunk = tokens_for_inference[i : i + INFERENCE_CHUNK_SIZE]
            
            # Prepare Tensors
            w_ids = torch.LongTensor([[word_vocab[t['word']] for t in chunk]]).to(DEVICE)
            
            c_ids_list = []
            for t in chunk:
                chars = [char_vocab[c] for c in t['word'][:MAX_CHAR_LEN]]
                chars += [0] * (MAX_CHAR_LEN - len(chars))
                c_ids_list.append(chars)
            c_ids = torch.LongTensor([c_ids_list]).to(DEVICE)
            
            bboxes = torch.FloatTensor([[[t['x0']/1000.0, t['y0']/1000.0, t['x1']/1000.0, t['y1']/1000.0] for t in chunk]]).to(DEVICE)
            s_feats = torch.FloatTensor([[t['spatial_features'] for t in chunk]]).to(DEVICE)
            c_feats = torch.FloatTensor([[t['context_features'] for t in chunk]]).to(DEVICE)
            mask = torch.ones(w_ids.size(), dtype=torch.bool).to(DEVICE)

            # Predict
            with torch.no_grad():
                preds = model(w_ids, c_ids, bboxes, s_feats, c_feats, mask)[0]
                
                # --- D. FORMAT OUTPUT ---
                for t, p in zip(chunk, preds):
                    label = IDX2LABEL[p]
                    # Create the exact dictionary structure expected by the rest of the pipeline
                    page_raw_predictions.append({
                        "word": t['word'],
                        "bbox": t['original_bbox'], # Pass back the scaled bbox the pipeline uses
                        "predicted_label": label,
                        "page_number": page_num
                    })

        if page_raw_predictions:
            final_page_predictions.append({
                "page_number": page_num,
                "data": page_raw_predictions
            })
            print(f"  -> Page {page_num} Inference Complete: {len(page_raw_predictions)} labeled words.")

    return final_page_predictions



# ============================================================================
# --- PHASE 3: BIO TO STRUCTURED JSON DECODER ---
# ============================================================================








def convert_bio_to_structured_json_relaxed(input_path: str, output_path: str) -> Optional[List[Dict[str, Any]]]:
    print("\n" + "=" * 80)
    print("--- 3. STARTING BIO TO STRUCTURED JSON DECODING ---")
    print(f"Source: {input_path}")
    print("=" * 80)
    
    start_time = time.time()

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            predictions_by_page = json.load(f)
        print(f"✅ Successfully loaded raw predictions ({len(predictions_by_page)} pages found)")
    except Exception as e:
        print(f"❌ Error loading raw prediction file: {e}")
        return None

    predictions = []
    for page_item in predictions_by_page:
        if isinstance(page_item, dict) and 'data' in page_item:
            predictions.extend(page_item['data'])
    
    total_words = len(predictions)
    print(f"📋 Total words to process: {total_words}")

    structured_data = []
    current_item = None
    current_option_key = None
    current_passage_buffer = []
    current_text_buffer = []
    first_question_started = False
    last_entity_type = None
    just_finished_i_option = False
    is_in_new_passage = False

    def finalize_passage_to_item(item, passage_buffer):
        if passage_buffer:
            passage_text = re.sub(r'\s{2,}', ' ', ' '.join(passage_buffer)).strip()
            print(f"   ↳ [Buffer] Finalizing passage ({len(passage_buffer)} words) into current item")
            if item.get('passage'):
                item['passage'] += ' ' + passage_text
            else:
                item['passage'] = passage_text
        passage_buffer.clear()

    # Iterate through every predicted word
    for idx, item in enumerate(predictions):
        word = item['word']
        label = item['predicted_label']
        entity_type = label[2:].strip() if label.startswith(('B-', 'I-')) else None
        current_text_buffer.append(word)
        
        previous_entity_type = last_entity_type
        is_passage_label = (entity_type == 'PASSAGE')

        # --- LOGGING: Track progress every 500 words or on B- labels ---
        if label.startswith('B-'):
             print(f"[Word {idx}/{total_words}] Found Label: {label} | Word: '{word}'")

        if not first_question_started:
            if label != 'B-QUESTION' and not is_passage_label:
                just_finished_i_option = False
                is_in_new_passage = False
                continue
            if is_passage_label:
                current_passage_buffer.append(word)
                last_entity_type = 'PASSAGE'
                just_finished_i_option = False
                is_in_new_passage = False
                continue

        if label == 'B-QUESTION':
            print(f"🔍 Detection: New Question Started at word {idx}")
            if not first_question_started:
                header_text = ' '.join(current_text_buffer[:-1]).strip()
                if header_text or current_passage_buffer:
                    print(f"   -> Creating METADATA item for text found before first question")
                    metadata_item = {'type': 'METADATA', 'passage': ''}
                    finalize_passage_to_item(metadata_item, current_passage_buffer)
                    if header_text: metadata_item['text'] = header_text
                    structured_data.append(metadata_item)
                first_question_started = True
                current_text_buffer = [word]

            if current_item is not None:
                finalize_passage_to_item(current_item, current_passage_buffer)
                current_item['text'] = ' '.join(current_text_buffer[:-1]).strip()
                structured_data.append(current_item)
                print(f"   -> Saved Question. Total structured items so far: {len(structured_data)}")
                current_text_buffer = [word]

            current_item = {
                'question': word, 'options': {}, 'answer': '', 'passage': '', 'text': ''
            }
            current_option_key = None
            last_entity_type = 'QUESTION'
            just_finished_i_option = False
            is_in_new_passage = False
            continue

        if current_item is not None:
            if is_in_new_passage:
                if 'new_passage' not in current_item:
                    current_item['new_passage'] = word
                else:
                    current_item['new_passage'] += f' {word}'
                
                if label.startswith('B-') or (label.startswith('I-') and entity_type != 'PASSAGE'):
                    print(f"   ↳ [State] Exiting new_passage mode at label {label}")
                    is_in_new_passage = False
                
                if label.startswith(('B-', 'I-')): 
                    last_entity_type = entity_type
                continue

            is_in_new_passage = False

            if label.startswith('B-'):
                if entity_type in ['QUESTION', 'OPTION', 'ANSWER', 'SECTION_HEADING']:
                    finalize_passage_to_item(current_item, current_passage_buffer)
                    current_passage_buffer = []
                
                last_entity_type = entity_type
                
                if entity_type == 'PASSAGE':
                    if previous_entity_type == 'OPTION' and just_finished_i_option:
                        print(f"   ↳ [State] Transitioning to new_passage (Option -> Passage boundary)")
                        current_item['new_passage'] = word
                        is_in_new_passage = True
                    else:
                        current_passage_buffer.append(word)
                
                elif entity_type == 'OPTION':
                    current_option_key = word
                    current_item['options'][current_option_key] = word
                    just_finished_i_option = False
                
                elif entity_type == 'ANSWER':
                    current_item['answer'] = word
                    current_option_key = None
                    just_finished_i_option = False
                
                elif entity_type == 'QUESTION':
                    current_item['question'] += f' {word}'
                    just_finished_i_option = False

            elif label.startswith('I-'):
                if entity_type == 'QUESTION':
                    current_item['question'] += f' {word}'
                elif entity_type == 'PASSAGE':
                    if previous_entity_type == 'OPTION' and just_finished_i_option:
                        current_item['new_passage'] = word
                        is_in_new_passage = True
                    else:
                        if not current_passage_buffer: last_entity_type = 'PASSAGE'
                        current_passage_buffer.append(word)
                elif entity_type == 'OPTION' and current_option_key is not None:
                    current_item['options'][current_option_key] += f' {word}'
                    just_finished_i_option = True
                elif entity_type == 'ANSWER':
                    current_item['answer'] += f' {word}'
                
                just_finished_i_option = (entity_type == 'OPTION')

            elif label == 'O':
                # if last_entity_type == 'QUESTION':
                #     current_item['question'] += f' {word}'
                # just_finished_i_option = False
                pass

    # Final wrap up
    if current_item is not None:
        print(f"🏁 Finalizing the very last item...")
        finalize_passage_to_item(current_item, current_passage_buffer)
        current_item['text'] = ' '.join(current_text_buffer).strip()
        structured_data.append(current_item)

    # Clean up and regex replacement
    for item in structured_data:
        item['text'] = re.sub(r'\s{2,}', ' ', item['text']).strip()
        if 'new_passage' in item:
            item['new_passage'] = re.sub(r'\s{2,}', ' ', item['new_passage']).strip()

    print(f"💾 Saving {len(structured_data)} items to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Decoding Complete. Total time: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"⚠️ Error saving final JSON: {e}")

    return structured_data

def create_query_text(entry: Dict[str, Any]) -> str:
    """Combines question and options into a single string for similarity matching."""
    query_parts = []
    if entry.get("question"):
        query_parts.append(entry["question"])

    for key in ["options", "options_text"]:
        options = entry.get(key)
        if options and isinstance(options, dict):
            for value in options.values():
                if value and isinstance(value, str):
                    query_parts.append(value)
    return " ".join(query_parts)


def calculate_similarity(doc1: str, doc2: str) -> float:
    """Calculates Cosine Similarity between two text strings."""
    if not doc1 or not doc2:
        return 0.0

    def clean_text(text):
        return re.sub(r'^\s*[\(\d\w]+\.?\s*', '', text, flags=re.MULTILINE)

    clean_doc1 = clean_text(doc1)
    clean_doc2 = clean_text(doc2)
    corpus = [clean_doc1, clean_doc2]

    try:
        vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern=r'(?u)\b\w\w+\b')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        if tfidf_matrix.shape[1] == 0:
            return 0.0
        vectors = tfidf_matrix.toarray()
        # Handle cases where vectors might be empty or too short
        if len(vectors) < 2:
            return 0.0
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return score
    except Exception:
        return 0.0






def process_context_linking(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Links questions to passages based on 'passage' flow vs 'new_passage' priority.
    Includes 'Decay Logic': If 2 consecutive questions fail to match the active passage,
    the passage context is dropped to prevent false positives downstream.
    """
    print("\n" + "=" * 80)
    print("--- STARTING CONTEXT LINKING (WITH DECAY LOGIC) ---")
    print("=" * 80)

    if not data: return []

    # --- PHASE 1: IDENTIFY PASSAGE DEFINERS ---
    passage_definer_indices = []
    for i, entry in enumerate(data):
        if entry.get("passage") and entry["passage"].strip():
            passage_definer_indices.append(i)
        if entry.get("new_passage") and entry["new_passage"].strip():
            if i not in passage_definer_indices:
                passage_definer_indices.append(i)

    # --- PHASE 2: CONTEXT TRANSFER & LINKING ---
    current_passage_text = None
    current_new_passage_text = None

    # NEW: Counter to track consecutive linking failures
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 2

    for i, entry in enumerate(data):
        item_type = entry.get("type", "Question")

        # A. UNCONDITIONALLY UPDATE CONTEXTS (And Reset Decay Counter)
        if entry.get("passage") and entry["passage"].strip():
            current_passage_text = entry["passage"]
            consecutive_failures = 0  # Reset because we have fresh explicit context
            # print(f"  [Flow] Updated Standard Context from Item {i}")

        if entry.get("new_passage") and entry["new_passage"].strip():
            current_new_passage_text = entry["new_passage"]
            # We don't necessarily reset standard failures here as this is a local override

        # B. QUESTION LINKING
        if entry.get("question") and item_type != "METADATA":
            combined_query = create_query_text(entry)

            # Skip if query is too short (noise)
            if len(combined_query.strip()) < 5:
                continue

            # Calculate scores
            score_old = calculate_similarity(current_passage_text, combined_query) if current_passage_text else 0.0
            score_new = calculate_similarity(current_new_passage_text,
                                             combined_query) if current_new_passage_text else 0.0

            # ------------------------------------------------------------------
            # 🛑 CRITICAL FIX APPLIED HERE 🛑
            # The original line: q_preview = entry['question'][:30] + '...'
            
            # 1. Capture the raw preview string (which might contain the bad surrogate)
            q_preview_raw = entry['question'][:30] + '...'
            
            # 2. Safely clean the string by encoding to UTF-8 and ignoring errors, 
            #    then decoding back. This removes the invalid surrogate character.
            q_preview = q_preview_raw.encode('utf-8', errors='ignore').decode('utf-8')
            # ------------------------------------------------------------------

            # RESOLUTION LOGIC
            linked = False

            # 1. Prefer New Passage if significantly better
            if current_new_passage_text and (score_new > score_old + RESOLUTION_MARGIN) and (
                    score_new >= SIMILARITY_THRESHOLD):
                entry["passage"] = current_new_passage_text
                print(f"  [Linker] 🚀 Q{i} ('{q_preview}') -> NEW PASSAGE (Score: {score_new:.3f})")
                linked = True
                # Note: We do not reset 'consecutive_failures' for the standard passage here,
                # because we matched the *new* passage, not the standard one.

            # 2. Otherwise use Standard Passage if it meets threshold
            elif current_passage_text and (score_old >= SIMILARITY_THRESHOLD):
                entry["passage"] = current_passage_text
                print(f"  [Linker] ✅ Q{i} ('{q_preview}') -> STANDARD PASSAGE (Score: {score_old:.3f})")
                linked = True
                consecutive_failures = 0  # Success! Reset the kill switch.

            if not linked:
                # 3. DECAY LOGIC
                if current_passage_text:
                    consecutive_failures += 1
                    # This is the line that was failing (or similar logging lines)
                    print(
                        f"  [Linker] ⚠️ Q{i} NOT LINKED. (Failures: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")

                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"  [Linker] 🗑️  Context dropped due to {consecutive_failures} consecutive misses.")
                        current_passage_text = None
                        consecutive_failures = 0
                else:
                    print(f"  [Linker] ⚠️ Q{i} NOT LINKED (No active context).")

    # --- PHASE 3: CLEANUP AND INTERPOLATION ---
    print("  [Linker] Running Cleanup & Interpolation...")

    # 3A. Self-Correction (Remove weak links)
    for i in passage_definer_indices:
        entry = data[i]
        if entry.get("question") and entry.get("type") != "METADATA":
            passage_to_check = entry.get("passage") or entry.get("new_passage")
            if passage_to_check:
                self_sim = calculate_similarity(passage_to_check, create_query_text(entry))
                if self_sim < SIMILARITY_THRESHOLD:
                    entry["passage"] = ""
                    if "new_passage" in entry: entry["new_passage"] = ""
                    print(f"  [Cleanup] Removed weak link for Q{i}")

    # 3B. Interpolation (Fill gaps)
    # We only interpolate if the gap is strictly 1 question wide to avoid undoing the decay logic
    for i in range(1, len(data) - 1):
        current_entry = data[i]
        is_gap = current_entry.get("question") and not current_entry.get("passage")
        if is_gap:
            prev_p = data[i - 1].get("passage")
            next_p = data[i + 1].get("passage")
            if prev_p and next_p and (prev_p == next_p) and prev_p.strip():
                current_entry["passage"] = prev_p
                print(f"  [Linker] 🥪 Q{i} Interpolated from neighbors.")

    return data







    


def correct_misaligned_options(structured_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("--- 5. STARTING POST-PROCESSING: OPTION ALIGNMENT CORRECTION ---")
    print("=" * 80)
    tag_pattern = re.compile(r'(EQUATION\d+|FIGURE\d+)')
    corrected_count = 0
    for item in structured_data:
        if item.get('type') in ['METADATA']: continue
        options = item.get('options')
        if not options or len(options) < 2: continue
        option_keys = list(options.keys())
        for i in range(len(option_keys) - 1):
            current_key = option_keys[i]
            next_key = option_keys[i + 1]
            current_value = options[current_key].strip()
            next_value = options[next_key].strip()
            is_current_empty = current_value == current_key
            content_in_next = next_value.replace(next_key, '', 1).strip()
            tags_in_next = tag_pattern.findall(content_in_next)
            has_two_tags = len(tags_in_next) == 2
            if is_current_empty and has_two_tags:
                tag_to_move = tags_in_next[0]
                options[current_key] = f"{current_key} {tag_to_move}".strip()
                options[next_key] = f"{next_key} {tags_in_next[1]}".strip()
                corrected_count += 1
    print(f"✅ Option alignment correction finished. Total corrections: {corrected_count}.")
    return structured_data



def get_base64_for_file(filepath: str) -> Optional[str]:
    """Reads a file and returns its Base64 encoded string without the data URI prefix."""
    try:
        with open(filepath, "rb") as image_file:
            # Return raw base64 string
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading and encoding file {filepath}: {e}")
        return None






def embed_images_as_base64_in_memory(structured_data: List[Dict[str, Any]], figure_extraction_dir: str) -> List[ Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("--- 4. STARTING IMAGE EMBEDDING (Base64) / EQUATION TO LATEX CONVERSION ---")
    print("=" * 80)
    if not structured_data:
        return []

    image_files = glob.glob(os.path.join(figure_extraction_dir, "*.png"))
    image_lookup = {}
    tag_regex = re.compile(r'(figure|equation)(\d+)', re.IGNORECASE)

    for filepath in image_files:
        filename = os.path.basename(filepath)
        match = re.search(r'_(figure|equation)(\d+)\.png$', filename, re.IGNORECASE)
        if match:
            key = f"{match.group(1).upper()}{match.group(2)}"
            image_lookup[key] = filepath

    print(f" -> Found {len(image_lookup)} image components.")
    
    final_structured_data = []

    for item in structured_data:
        text_fields = [item.get('question', ''), item.get('passage', '')]
        if 'options' in item:
            for opt_val in item['options'].values():
                text_fields.append(opt_val)
        if 'new_passage' in item:
            text_fields.append(item['new_passage'])

        unique_tags_to_embed = set()
        for text in text_fields:
            if not text: continue
            for match in tag_regex.finditer(text):
                tag = match.group(0).upper()
                if tag in image_lookup:
                    unique_tags_to_embed.add(tag)

        # List of tags that were successfully converted to LaTeX
        tags_converted_to_latex = set()
        
        for tag in sorted(list(unique_tags_to_embed)):
            filepath = image_lookup[tag]
            base_key = tag.replace(' ', '').lower() # e.g., figure1 or equation1

            if 'EQUATION' in tag:
                # Equation to LaTeX conversion
                base64_code = get_base64_for_file(filepath) # This reads the file for conversion
                if base64_code:
                    latex_output = get_latex_from_base64(base64_code)
                    if not latex_output.startswith('[P2T_ERROR') and not latex_output.startswith('[P2T_WARNING'):
                        # *** CORE CHANGE: Store the clean LaTeX output directly ***
                        item[base_key] = latex_output
                        tags_converted_to_latex.add(tag)
                        print(f"  ✅ Embedded Clean LaTeX for {tag}")
                    else:
                        # On failure, embed the error message
                        item[base_key] = latex_output
                        print(f"  ⚠️ Failed to convert {tag} to LaTeX. Embedding error message.")
                else:
                     item[base_key] = "[FILE_ERROR: Could not read image file]"
                     print(f"  ❌ File read error for {tag}.")
                     
            elif 'FIGURE' in tag:
                # Figure to Base64 conversion
                base64_code = get_base64_for_file(filepath)
                item[base_key] = base64_code
                print(f"  ✅ Embedded Base64 for {tag}")

        final_structured_data.append(item)

    print(f"✅ Image embedding complete.")
    return final_structured_data









# ============================================================================
# --- MAIN FUNCTION ---
# ============================================================================






        




def classify_question_type(item: Dict[str, Any]) -> str:
    """
    Classifies a question as 'MCQ', 'DESCRIPTIVE', or 'INTEGER' based on its options.
    
    Args:
        item: Dictionary containing question data with 'options' field
        
    Returns:
        str: 'MCQ' if options exist and are non-empty, 'DESCRIPTIVE' otherwise
    """
    # Check if options exist and have meaningful content
    options = item.get('options', {})
    
    if not options:
        return 'DESCRIPTIVE'
    
    # Check if options dict has keys and at least one non-empty value
    has_valid_options = False
    for key, value in options.items():
        # Check if the value is more than just the key itself (e.g., "A" vs "A Some text")
        if value and isinstance(value, str):
            # Remove the key from value and check if there's remaining content
            remaining_text = value.replace(key, '').strip()
            if remaining_text and len(remaining_text) > 0:
                has_valid_options = True
                break
    
    return 'MCQ' if has_valid_options else 'DESCRIPTIVE'


def add_question_type_validation(structured_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds 'question_type' field to all question entries in the structured data.
    
    Args:
        structured_data: List of dictionaries containing question data
        
    Returns:
        List[Dict[str, Any]]: Modified list with 'question_type' field added
    """
    print("\n" + "=" * 80)
    print("--- ADDING QUESTION TYPE VALIDATION ---")
    print("=" * 80)
    
    mcq_count = 0
    descriptive_count = 0
    metadata_count = 0
    
    for item in structured_data:
        item_type = item.get('type', 'Question')
        
        # Skip metadata entries
        if item_type == 'METADATA':
            metadata_count += 1
            item['question_type'] = 'METADATA'
            continue
        
        # Classify the question
        question_type = classify_question_type(item)
        item['question_type'] = question_type
        
        if question_type == 'MCQ':
            mcq_count += 1
        else:
            descriptive_count += 1
    
    print(f"  ✅ Classification Complete:")
    print(f"     - MCQ Questions: {mcq_count}")
    print(f"     - Descriptive/Integer Questions: {descriptive_count}")
    print(f"     - Metadata Entries: {metadata_count}")
    print(f"     - Total Entries: {len(structured_data)}")
    
    return structured_data




import time
import traceback
import glob







# def run_document_pipeline(input_pdf_path: str, layoutlmv3_model_path: str, structured_intermediate_output_path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
#     if not os.path.exists(input_pdf_path): 
#         print(f"❌ ERROR: File not found: {input_pdf_path}")
#         return None

#     print("\n" + "#" * 80)
#     print("### STARTING OPTIMIZED FULL DOCUMENT ANALYSIS PIPELINE ###")
#     print(f"Input: {input_pdf_path}")
#     print("#" * 80)

#     overall_start = time.time()
#     pdf_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
#     temp_pipeline_dir = os.path.join(tempfile.gettempdir(), f"pipeline_run_{pdf_name}_{os.getpid()}")
#     os.makedirs(temp_pipeline_dir, exist_ok=True)

#     preprocessed_json_path = os.path.join(temp_pipeline_dir, f"{pdf_name}_preprocessed.json")
#     raw_output_path = os.path.join(temp_pipeline_dir, f"{pdf_name}_raw_predictions.json")

#     if structured_intermediate_output_path is None:
#         structured_intermediate_output_path = os.path.join(
#             temp_pipeline_dir, f"{pdf_name}_structured_intermediate.json"
#         )
    
#     final_result = None
#     try:
#         # --- Phase 1: Preprocessing ---
#         print(f"\n[Step 1/5] Preprocessing (YOLO + Masking)...")
#         p1_start = time.time()
#         preprocessed_json_path_out = run_single_pdf_preprocessing(input_pdf_path, preprocessed_json_path)
#         if not preprocessed_json_path_out:
#             print("❌ FAILED at Step 1: Preprocessing returned None.")
#             return None
#         print(f"✅ Step 1 Complete ({time.time() - p1_start:.2f}s)")

#         # --- Phase 2: Inference ---
#         print(f"\n[Step 2/5] Inference (LayoutLMv3)...")
#         p2_start = time.time()
#         page_raw_predictions_list = run_inference_and_get_raw_words(
#             input_pdf_path, layoutlmv3_model_path, preprocessed_json_path_out
#         )
#         if not page_raw_predictions_list:
#             print("❌ FAILED at Step 2: Inference returned no data.")
#             return None
        
#         with open(raw_output_path, 'w', encoding='utf-8') as f:
#             json.dump(page_raw_predictions_list, f, indent=4)
#         print(f"✅ Step 2 Complete ({time.time() - p2_start:.2f}s)")

#         # --- Phase 3: Decoding ---
#         print(f"\n[Step 3/5] Decoding (BIO to Structured JSON)...")
#         p3_start = time.time()
#         structured_data_list = convert_bio_to_structured_json_relaxed(
#             raw_output_path, structured_intermediate_output_path
#         )
#         if not structured_data_list:
#             print("❌ FAILED at Step 3: BIO conversion failed.")
#             return None

#         print("... Correcting misalignments and linking context ...")
#         structured_data_list = correct_misaligned_options(structured_data_list)
#         structured_data_list = process_context_linking(structured_data_list)
#         print(f"✅ Step 3 Complete ({time.time() - p3_start:.2f}s)")

#         # --- Phase 4: Base64 & LaTeX ---
#         print(f"\n[Step 4/5] Finalizing Layout (Base64 Images & LaTeX)...")
#         p4_start = time.time()
#         final_result = embed_images_as_base64_in_memory(structured_data_list, FIGURE_EXTRACTION_DIR)
#         if not final_result:
#             print("❌ FAILED at Step 4: Final formatting failed.")
#             return None
#         print(f"✅ Step 4 Complete ({time.time() - p4_start:.2f}s)")

#         # --- Phase 4.5: Question Type Classification ---
#         print(f"\n[Step 4.5/5] Adding Question Type Classification...")
#         p4_5_start = time.time()
#         final_result = add_question_type_validation(final_result)
#         print(f"✅ Step 4.5 Complete ({time.time() - p4_5_start:.2f}s)")

#         # --- Phase 5: Hierarchical Tagging ---
#         print(f"\n[Step 5/5] AI Classification (Subject/Concept Tagging)...")
#         p5_start = time.time()
#         classifier = HierarchicalClassifier()
#         if classifier.load_models():
#             final_result = post_process_json_with_inference(final_result, classifier)
#             print(f"✅ Step 5 Complete: Tags added ({time.time() - p5_start:.2f}s)")
#         else:
#             print("⚠️ WARNING: Classifier models failed to load. Skipping tags.")

#         # ============================================================
#         # 🔧 NEW STEP: FILTER OUT METADATA ENTRIES
#         # ============================================================
#         print(f"\n[Post-Processing] Removing METADATA entries...")
#         initial_count = len(final_result)
#         final_result = [item for item in final_result if item.get('type') != 'METADATA']
#         removed_count = initial_count - len(final_result)
#         print(f"✅ Removed {removed_count} METADATA entries. {len(final_result)} questions remain.")
#         # ============================================================

#     except Exception as e:
#         print(f"\n‼️ FATAL PIPELINE EXCEPTION:")
#         print(f"Error Message: {str(e)}")
#         traceback.print_exc()
#         return None

#     # finally:
#     #     print(f"\nCleaning up temporary files in {temp_pipeline_dir}...")
#     #     try:
#     #         for f in glob.glob(os.path.join(temp_pipeline_dir, '*')):
#     #             os.remove(f)
#     #         os.rmdir(temp_pipeline_dir)
#     #         print("🧹 Cleanup successful.")
#     #     except Exception as e:
#     #         print(f"⚠️ Cleanup failed: {e}")

#     total_time = time.time() - overall_start
#     print("\n" + "#" * 80)
#     print(f"### PIPELINE COMPLETE | Total Time: {total_time:.2f}s ###")
#     print("#" * 80)
    
#     return final_result



def run_document_pipeline(input_pdf_path: str, layoutlmv3_model_path: str, structured_intermediate_output_path: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(input_pdf_path): 
        print(f"❌ ERROR: File not found: {input_pdf_path}")
        return None

    print("\n" + "#" * 80)
    print("### STARTING OPTIMIZED FULL DOCUMENT ANALYSIS PIPELINE ###")
    print(f"Input: {input_pdf_path}")
    print("#" * 80)

    overall_start = time.time()
    pdf_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
    temp_pipeline_dir = os.path.join(tempfile.gettempdir(), f"pipeline_run_{pdf_name}_{os.getpid()}")
    os.makedirs(temp_pipeline_dir, exist_ok=True)

    preprocessed_json_path = os.path.join(temp_pipeline_dir, f"{pdf_name}_preprocessed.json")
    raw_output_path = os.path.join(temp_pipeline_dir, f"{pdf_name}_raw_predictions.json")

    if structured_intermediate_output_path is None:
        structured_intermediate_output_path = os.path.join(
            temp_pipeline_dir, f"{pdf_name}_structured_intermediate.json"
        )
    
    final_result = None
    try:
        # --- Phase 1: Preprocessing ---
        print(f"\n[Step 1/5] Preprocessing (YOLO + Masking)...")
        p1_start = time.time()
        preprocessed_json_path_out = run_single_pdf_preprocessing(input_pdf_path, preprocessed_json_path)
        if not preprocessed_json_path_out:
            print("❌ FAILED at Step 1: Preprocessing returned None.")
            return None
        print(f"✅ Step 1 Complete ({time.time() - p1_start:.2f}s)")

        # --- Phase 2: Inference (MODIFIED) ---
        print(f"\n[Step 2/5] Inference (Custom Model)...")
        p2_start = time.time()
        
        # -------------------------------------------------------------------------
        # --- COMMENTED OUT OLD LAYOUTLMV3 CALL FOR REVERSION ---
        page_raw_predictions_list = run_inference_and_get_raw_words(
           input_pdf_path, layoutlmv3_model_path, preprocessed_json_path_out
        )
        # -------------------------------------------------------------------------

        # --- NEW CUSTOM MODEL CALL ---
        # Note: We only pass the JSON path because the custom function 
        # doesn't need to re-read the PDF or use the layoutlmv3 model path.
        # page_raw_predictions_list = run_custom_inference_and_get_raw_words(
        #     preprocessed_json_path_out
        # )
        # -----------------------------

        if not page_raw_predictions_list:
            print("❌ FAILED at Step 2: Inference returned no data.")
            return None
        
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            json.dump(page_raw_predictions_list, f, indent=4)
        print(f"✅ Step 2 Complete ({time.time() - p2_start:.2f}s)")

        # --- Phase 3: Decoding ---
        print(f"\n[Step 3/5] Decoding (BIO to Structured JSON)...")
        p3_start = time.time()
        structured_data_list = convert_bio_to_structured_json_relaxed(
            raw_output_path, structured_intermediate_output_path
        )
        if not structured_data_list:
            print("❌ FAILED at Step 3: BIO conversion failed.")
            return None

        print("... Correcting misalignments and linking context ...")
        structured_data_list = correct_misaligned_options(structured_data_list)
        structured_data_list = process_context_linking(structured_data_list)
        print(f"✅ Step 3 Complete ({time.time() - p3_start:.2f}s)")

        # --- Phase 4: Base64 & LaTeX ---
        print(f"\n[Step 4/5] Finalizing Layout (Base64 Images & LaTeX)...")
        p4_start = time.time()
        final_result = embed_images_as_base64_in_memory(structured_data_list, FIGURE_EXTRACTION_DIR)
        if not final_result:
            print("❌ FAILED at Step 4: Final formatting failed.")
            return None
        print(f"✅ Step 4 Complete ({time.time() - p4_start:.2f}s)")

        # --- Phase 4.5: Question Type Classification ---
        print(f"\n[Step 4.5/5] Adding Question Type Classification...")
        p4_5_start = time.time()
        final_result = add_question_type_validation(final_result)
        print(f"✅ Step 4.5 Complete ({time.time() - p4_5_start:.2f}s)")

        # --- Phase 5: Hierarchical Tagging ---
        print(f"\n[Step 5/5] AI Classification (Subject/Concept Tagging)...")
        p5_start = time.time()
        classifier = HierarchicalClassifier()
        if classifier.load_models():
            final_result = post_process_json_with_inference(final_result, classifier)
            print(f"✅ Step 5 Complete: Tags added ({time.time() - p5_start:.2f}s)")
        else:
            print("⚠️ WARNING: Classifier models failed to load. Skipping tags.")

        # ============================================================
        # 🔧 NEW STEP: FILTER OUT METADATA ENTRIES
        # ============================================================
        print(f"\n[Post-Processing] Removing METADATA entries...")
        initial_count = len(final_result)
        final_result = [item for item in final_result if item.get('type') != 'METADATA']
        removed_count = initial_count - len(final_result)
        print(f"✅ Removed {removed_count} METADATA entries. {len(final_result)} questions remain.")
        # ============================================================

    except Exception as e:
        print(f"\n‼️ FATAL PIPELINE EXCEPTION:")
        print(f"Error Message: {str(e)}")
        traceback.print_exc()
        return None

    # finally:
    #     print(f"\nCleaning up temporary files in {temp_pipeline_dir}...")
    #     try:
    #         for f in glob.glob(os.path.join(temp_pipeline_dir, '*')):
    #             os.remove(f)
    #         os.rmdir(temp_pipeline_dir)
    #         print("🧹 Cleanup successful.")
    #     except Exception as e:
    #         print(f"⚠️ Cleanup failed: {e}")

    total_time = time.time() - overall_start
    print("\n" + "#" * 80)
    print(f"### PIPELINE COMPLETE | Total Time: {total_time:.2f}s ###")
    print("#" * 80)
    
    return final_result





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete Pipeline")
    parser.add_argument("--input_pdf", type=str, required=True, help="Input PDF")
    parser.add_argument("--layoutlmv3_model_path", type=str, default=DEFAULT_LAYOUTLMV3_MODEL_PATH, help="Model Path")

    # --- ADDED ARGUMENT FOR DEBUGGING ---
    parser.add_argument("--raw_preds_path", type=str, default='BIO_debug.json',
                        help="Debug path for raw BIO tag predictions (JSON).")
    # ------------------------------------
    args = parser.parse_args()

    pdf_name = os.path.splitext(os.path.basename(args.input_pdf))[0]
    final_output_path = os.path.abspath(f"{pdf_name}_final_output_embedded.json")

    # --- CALCULATE RAW PREDICTIONS OUTPUT PATH (Kept commented as per original script) ---
    # raw_predictions_output_path = os.path.abspath(
    #     args.raw_preds_path if args.raw_preds_path else f"{pdf_name}_raw_predictions_debug.json")
    # ---------------------------------------------

    # --- UPDATED FUNCTION CALL ---
    final_json_data = run_document_pipeline(
        args.input_pdf,
        args.layoutlmv3_model_path )
    # -----------------------------

    # 🛑 CRITICAL FINAL FIX: AGGRESSIVE CUSTOM JSON SAVING 🛑
    if final_json_data:
        # 1. Dump the Python object to a standard JSON string.
        # This converts the in-memory double backslash ('\\') into a quadruple backslash ('\\\\')
        # in the raw json_str string content.
        json_str = json.dumps(final_json_data, indent=2, ensure_ascii=False)

        # 2. **AGGRESSIVE UNDO ESCAPING:** We assume we have quadruple backslashes and
        # replace them with the double backslashes needed for the LaTeX command to work.
        # This operation essentially replaces four literal backslashes with two literal backslashes.
        # final_output_content = json_str.replace('\\\\\\\\', '\\\\')

        # 3. Write the corrected string content to the file.
        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
            
        print(f"\n✅ Final Data Saved: {final_output_path}")
    else:
        print("\n❌ Pipeline Failed.")
        sys.exit(1)
