import json
import argparse
import os
import torch
import torch.nn as nn
from TorchCRF import CRF
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Model
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import sys
import io

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ============================================================================
# COLUMN DETECTION MODULE (INTEGRATED)
# ============================================================================

def get_word_data_for_detection(page: fitz.Page, top_margin_percent=0.10, bottom_margin_percent=0.10) -> list:
    """Extracts word data for column detection with Y-axis filtering."""
    word_data = page.get_text("words")

    if len(word_data) > 0:
        full_word_data = [(w[4], w[0], w[1], w[2], w[3]) for w in word_data]
    else:
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))

            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            ocr_words = []
            scale_x = page.rect.width / img.width
            scale_y = page.rect.height / img.height

            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                try:
                    conf_val = float(data['conf'][i])
                except Exception:
                    conf_val = -1.0

                if text and int(data['level'][i]) == 5 and conf_val > 60:
                    x = data['left'][i] * scale_x
                    y = data['top'][i] * scale_y
                    w = data['width'][i] * scale_x
                    h = data['height'][i] * scale_y
                    x0, y0, x1, y1 = x, y, x + w, y + h
                    ocr_words.append((text, x0, y0, x1, y1))

            full_word_data = ocr_words
        except Exception as e:
            return []

    if not full_word_data:
        return []

    page_height = page.rect.height
    Y_TOP_LIMIT = page_height * top_margin_percent
    Y_BOTTOM_LIMIT = page_height * (1.0 - bottom_margin_percent)

    filtered_words = [
        (text, x0, y0, x1, y1) for text, x0, y0, x1, y1 in full_word_data
        if y0 >= Y_TOP_LIMIT and y1 <= Y_BOTTOM_LIMIT
    ]

    return filtered_words


def detect_columns_density_based(word_data: list, page_width: float, page_height: float,
                                 density_resolution=2, density_smoothing=3,
                                 gutter_prominence=0.60, gutter_min_width=5,
                                 gutter_min_distance=70,
                                 gutter_relative_prominence=0.70) -> list:
    """Detects column boundaries using horizontal density analysis."""
    if not word_data or len(word_data) < 10:
        return []

    resolution = density_resolution
    num_bins = max(1, int(page_width / resolution))
    density = np.zeros(num_bins)

    for text, x0, y0, x1, y1 in word_data:
        start_bin = int(x0 / resolution)
        end_bin = int(x1 / resolution)
        start_bin = max(0, min(start_bin, num_bins - 1))
        end_bin = max(0, min(end_bin, num_bins - 1))
        weight = (y1 - y0)
        density[start_bin:end_bin + 1] += weight

    density_smooth = gaussian_filter1d(density, sigma=density_smoothing)
    inverted_density = np.max(density_smooth) - density_smooth
    distance_in_bins = max(1, int(gutter_min_distance / resolution))

    try:
        peaks, properties = find_peaks(
            inverted_density,
            prominence=np.max(inverted_density) * gutter_prominence,
            width=gutter_min_width,
            distance=distance_in_bins
        )
    except Exception:
        return []

    prominences = properties.get('prominences', np.array([]))
    if len(prominences) > 0:
        max_prominence = np.max(prominences)
        relative_threshold = max_prominence * gutter_relative_prominence
        filtered_peaks = peaks[prominences >= relative_threshold]
    else:
        filtered_peaks = peaks

    gutters = [peak * resolution for peak in filtered_peaks]
    return sorted(gutters)


def detect_columns_clustering(word_data: list, page_width: float,
                              cluster_bin_size=5, cluster_smoothing=2,
                              cluster_threshold_percentile=30, cluster_min_width=20) -> list:
    """Detects columns by clustering word x-positions."""
    if not word_data or len(word_data) < 10:
        return []

    x_centers = [(x0 + x1) / 2 for _, x0, y0, x1, y1 in word_data]
    num_bins = max(1, int(page_width / cluster_bin_size))
    hist, bin_edges = np.histogram(x_centers, bins=num_bins, range=(0, page_width))
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=cluster_smoothing)

    non_zero_hist = hist_smooth[hist_smooth > 0]
    if len(non_zero_hist) == 0:
        threshold = 0
    else:
        threshold = np.percentile(non_zero_hist, cluster_threshold_percentile)

    gutters = []
    in_gutter = False
    gutter_start_x = 0.0

    for i, count in enumerate(hist_smooth):
        x_pos = bin_edges[i]
        if count < threshold and not in_gutter:
            in_gutter = True
            gutter_start_x = x_pos
        elif count >= threshold and in_gutter:
            in_gutter = False
            gutter_end_x = x_pos
            if (gutter_end_x - gutter_start_x) > cluster_min_width:
                gutter_center = (gutter_start_x + gutter_end_x) / 2
                gutters.append(gutter_center)

    return sorted(gutters)




def detect_column_gutters(pdf_path: str, page_num: int, **kwargs) -> Optional[int]:
    """
    Detects column gutters for a specific page using advanced algorithms.
    Returns the X-coordinate of the primary gutter, or None if single column.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        top_margin = kwargs.get('top_margin_percent', 0.10)
        bottom_margin = kwargs.get('bottom_margin_percent', 0.10)
        
        word_data = get_word_data_for_detection(page, top_margin, bottom_margin)
        page_width = page.rect.width
        page_height = page.rect.height
        
        if not word_data:
            doc.close()
            return None

        # Strategy 1: Density-based detection
        gutters_density = detect_columns_density_based(
            word_data, page_width, page_height,
            kwargs.get('density_resolution', 2),
            kwargs.get('density_smoothing', 3),
            kwargs.get('gutter_prominence', 0.15),
            kwargs.get('gutter_min_width', 5),
            kwargs.get('gutter_min_distance', 50),
            kwargs.get('gutter_relative_prominence', 0.50)
        )

        # Strategy 2: Clustering-based detection
        gutters_clustering = detect_columns_clustering(
            word_data, page_width,
            kwargs.get('cluster_bin_size', 5),
            kwargs.get('cluster_smoothing', 2),
            kwargs.get('cluster_threshold_percentile', 20),
            kwargs.get('cluster_min_width', 20)
        )

        # Combine results
        all_gutters = gutters_density + gutters_clustering
        if not all_gutters:
            doc.close()
            return None

        # Merge nearby gutters
        merge_tolerance = kwargs.get('merge_tolerance', 30)
        all_gutters.sort()
        final_gutters = []
        current_cluster = [all_gutters[0]]

        for gutter in all_gutters[1:]:
            if gutter - current_cluster[-1] <= merge_tolerance:
                current_cluster.append(gutter)
            else:
                final_gutters.append(np.median(current_cluster))
                current_cluster = [gutter]

        if current_cluster:
            final_gutters.append(np.median(current_cluster))

        # Filter edge gutters
        edge_margin = kwargs.get('edge_margin_percent', 0.05)
        margin = page_width * edge_margin
        final_gutters = [g for g in final_gutters if margin < g < (page_width - margin)]

        # Central gutter filter
        central_min = page_width * 0.4
        central_max = page_width * 0.6
        final_gutters = [g for g in final_gutters if central_min <= g <= central_max]

        doc.close()
        
        # Return only the first (most prominent) gutter as a single integer
        if final_gutters:
            return int(final_gutters[0])
        else:
            return None
        
    except Exception as e:
        print(f"Error in column detection: {e}")
        return None
    


def split_data_by_gutter(words: List[str], bboxes_raw: List[List[int]],
                        normalized_bboxes: List[List[int]], separator_x: int,
                        page_width: int, page_height: int) -> List[Dict[str, Any]]:
    """
    Splits word data into two columns based on gutter position and rebases second column.
    """
    split_data = []
    col1_words, col1_bboxes_raw, col1_normalized_bboxes = [], [], []
    col2_words, col2_bboxes_raw, col2_normalized_bboxes = [], [], []

    for word, bbox_raw, bbox_norm in zip(words, bboxes_raw, normalized_bboxes):
        x0_raw = bbox_raw[0]

        if x0_raw < separator_x:
            col1_words.append(word)
            col1_bboxes_raw.append(bbox_raw)
            col1_normalized_bboxes.append(bbox_norm)
        else:
            # Rebase Raw BBox
            new_x0_raw = max(0, bbox_raw[0] - separator_x)
            new_x1_raw = max(0, bbox_raw[2] - separator_x)
            new_bbox_raw = [new_x0_raw, bbox_raw[1], new_x1_raw, bbox_raw[3]]

            # Rebase Normalized BBox
            new_bbox_norm = [
                int(1000 * new_bbox_raw[0] / page_width),
                int(1000 * new_bbox_raw[1] / page_height),
                int(1000 * new_bbox_raw[2] / page_width),
                int(1000 * new_bbox_raw[3] / page_height)
            ]
            new_bbox_norm = [min(1000, max(0, b)) for b in new_bbox_norm]

            col2_words.append(word)
            col2_bboxes_raw.append(new_bbox_raw)
            col2_normalized_bboxes.append(new_bbox_norm)

    if col1_words:
        split_data.append({
            "words": col1_words,
            "bboxes_raw": col1_bboxes_raw,
            "normalized_bboxes": col1_normalized_bboxes
        })

    if col2_words:
        split_data.append({
            "words": col2_words,
            "bboxes_raw": col2_bboxes_raw,
            "normalized_bboxes": col2_normalized_bboxes
        })

    return split_data


# ============================================================================
# ORIGINAL PIPELINE COMPONENTS
# ============================================================================

# --- 1. Model Architecture ---
class LayoutLMv3CRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.layoutlm = LayoutLMv3Model.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.layoutlm.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, bbox, attention_mask, labels=None):
        outputs = self.layoutlm(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=attention_mask.bool())
            return -log_likelihood.mean()
        else:
            best_paths = self.crf.viterbi_decode(emissions, mask=attention_mask.bool())
            return best_paths


# --- 2. OCR Fallback ---
def run_tesseract_ocr(img: Image, page_width, page_height) -> tuple[list, list, list]:
    """Helper to run Tesseract and normalize bounding boxes."""
    tesseract_config = '--psm 6'

    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=tesseract_config)
    valid_indices = [i for i, word in enumerate(ocr_data['text']) if word.strip()]

    words = [ocr_data['text'][i] for i in valid_indices]

    x_scale = page_width / img.width
    y_scale = page_height / img.height

    bboxes_raw = []
    for i in valid_indices:
        left = ocr_data['left'][i]
        top = ocr_data['top'][i]
        width = ocr_data['width'][i]
        height = ocr_data['height'][i]

        raw_bbox = [
            left * x_scale,
            top * y_scale,
            (left + width) * x_scale,
            (top + height) * y_scale
        ]
        bboxes_raw.append([int(b) for b in raw_bbox])

    normalized_bboxes = [[
        int(1000 * b[0] / page_width),
        int(1000 * b[1] / page_height),
        int(1000 * b[2] / page_width),
        int(1000 * b[3] / page_height)
    ] for b in bboxes_raw]

    return words, bboxes_raw, normalized_bboxes


# --- 3. Helper for Label Studio Output ---
def create_label_studio_span(all_results, start_idx, end_idx, label):
    """Create a Label Studio span with character-level offsets."""
    entity_words = [all_results[i]['word'] for i in range(start_idx, end_idx + 1)]
    entity_bboxes = [all_results[i]['bbox'] for i in range(start_idx, end_idx + 1)]

    x0 = min(bbox[0] for bbox in entity_bboxes)
    y0 = min(bbox[1] for bbox in entity_bboxes)
    x1 = max(bbox[2] for bbox in entity_bboxes)
    y1 = max(bbox[3] for bbox in entity_bboxes)

    all_words = [r['word'] for r in all_results]
    text_string = " ".join(all_words)

    prefix_words = all_words[:start_idx]
    start_char = len(" ".join(prefix_words)) + (1 if prefix_words else 0)
    span_text = " ".join(entity_words)
    end_char = start_char + len(span_text)

    return {
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {
            "start": start_char,
            "end": end_char,
            "text": span_text,
            "labels": [label],
            "bbox": {
                "x": x0,
                "y": y0,
                "width": x1 - x0,
                "height": y1 - y0
            }
        },
        "score": 0.99
    }


# # --- 4. Core Inference Function (MODIFIED WITH NEW COLUMN DETECTION) ---


# def run_inference_and_structure(pdf_path: str, model_path: str, inference_output_path: str, 
#                                column_detection_params: Dict = None) -> List[Dict[str, Any]]:
#     labels = [
#         "O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION",
#         "B-ANSWER", "I-ANSWER", "B-SECTION_HEADING", "I-SECTION_HEADING",
#         "B-PASSAGE", "I-PASSAGE"
#     ]

#     if column_detection_params is None:
#         column_detection_params = {
#             'edge_margin_percent': 0.05,
#             'merge_tolerance': 30,
#             'top_margin_percent': 0.10,
#             'bottom_margin_percent': 0.10,
#             'density_resolution': 2,
#             'density_smoothing': 3,
#             'gutter_prominence': 0.15,
#             'gutter_min_width': 5,
#             'gutter_min_distance': 50,
#             'gutter_relative_prominence': 0.50,
#             'cluster_bin_size': 5,
#             'cluster_smoothing': 2,
#             'cluster_threshold_percentile': 20,
#             'cluster_min_width': 20,
#         }

#     id2label = {i: l for i, l in enumerate(labels)}
#     tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LayoutLMv3CRF("microsoft/layoutlmv3-base", num_labels=len(labels)).to(device)

#     if not os.path.exists(model_path):
#         print(f"❌ Error: Model checkpoint not found at {model_path}. Please check the path.")
#         return []

#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     print(f"✅ Model loaded successfully from {model_path} with {len(labels)} labels.")

#     try:
#         doc = pdfplumber.open(pdf_path)
#     except Exception as e:
#         print(f"❌ Error opening PDF with pdfplumber: {e}")
#         return []

#     all_pages_data = []
#     CHUNK_SIZE = 500

#     for page_num, page in enumerate(doc.pages):
#         print(f"\nProcessing page {page_num + 1}...")

#         page_width, page_height = page.width, page.height
#         words = []
#         bboxes_raw = []
#         normalized_bboxes = []

#         # --- Word Extraction (PDFPlumber) ---
#         word_list = page.extract_words(x_tolerance=1, y_tolerance=1, keep_blank_chars=False, use_text_flow=True)
#         if word_list:
#             print(f"  (Page {page_num + 1}) Found {len(word_list)} words using pdfplumber.")
#             for word_data in word_list:
#                 word = word_data['text']
#                 raw_bbox = [word_data['x0'], word_data['top'], word_data['x1'], word_data['bottom']]

#                 words.append(word)
#                 bboxes_raw.append([int(b) for b in raw_bbox])

#                 normalized_bboxes.append([
#                     max(0, min(1000, int(1000 * raw_bbox[0] / page_width))),
#                     max(0, min(1000, int(1000 * raw_bbox[1] / page_height))),
#                     max(0, min(1000, int(1000 * raw_bbox[2] / page_width))),
#                     max(0, min(1000, int(1000 * raw_bbox[3] / page_height)))
#                 ])

#         # --- OCR Fallback ---
#         if not words:
#             print(f"  (Page {page_num + 1}) No text layer found. Running Tesseract OCR fallback...")
#             try:
#                 fitz_doc = fitz.open(pdf_path)
#                 fitz_page = fitz_doc.load_page(page_num)
#                 pix = fitz_page.get_pixmap(dpi=300)
#                 img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#                 fitz_doc.close()
#                 words, bboxes_raw, normalized_bboxes = run_tesseract_ocr(img, page_width, page_height)
#             except Exception as e:
#                 print(f"  (Page {page_num + 1}) Error during OCR fallback: {e}")

#             if not words:
#                 print(f"  (Page {page_num + 1}) OCR failed to find any words. Skipping page.")
#                 continue

#         # --- NEW COLUMN DETECTION INTEGRATION ---
#         separator_x = detect_column_gutters(pdf_path, page_num, **column_detection_params)
        
#         # Force conversion to int if it's a list (safety check)
#         if separator_x is not None:
#             if isinstance(separator_x, list):
#                 separator_x = int(separator_x[0]) if separator_x else None
        
#         if separator_x is not None:
#             print(f"  (Page {page_num + 1}) ✅ Column separator detected at X={separator_x}")
#             split_data = split_data_by_gutter(words, bboxes_raw, normalized_bboxes, 
#                                              separator_x, page_width, page_height)
#         else:
#             print(f"  (Page {page_num + 1}) Single column structure assumed.")
#             split_data = []

#         # If no split occurred, treat as single column
#         if not split_data:
#             split_data.append({
#                 "words": words,
#                 "bboxes_raw": bboxes_raw,
#                 "normalized_bboxes": normalized_bboxes
#             })

#         # Apply preprocessing to EACH column (whether split or not)
#         for col_idx, col_data in enumerate(split_data):
#             print(f"  DEBUG: Preprocessing column {col_idx + 1}, has {len(col_data['words'])} words")
#             col_words = col_data['words']
#             col_bboxes_raw = col_data['bboxes_raw']
#             col_normalized_bboxes = col_data['normalized_bboxes']

#             if col_normalized_bboxes:
#              min_x_min = min(bbox[0] for bbox in col_normalized_bboxes)
#              if min_x_min < 110:
#               difference = min_x_min - 110
        
#              # Adjust normalized bboxes
#               new_normalized_bboxes = []
#               for bbox in col_normalized_bboxes:
#                 new_bbox = [
#                   bbox[0] - difference, bbox[1],
#                   bbox[2] - difference, bbox[3]
#                 ]
#                 new_normalized_bboxes.append([min(1000, max(0, b)) for b in new_bbox])
#             col_normalized_bboxes = new_normalized_bboxes
        
#             # ALSO adjust raw bboxes proportionally
#             raw_difference = difference * page_width / 1000.0  # Convert normalized difference to raw pixels
#               new_bboxes_raw = []
#                for bbox_raw in col_bboxes_raw:
#                 new_bbox_raw = [
#                    int(bbox_raw[0] - raw_difference),
#                    bbox_raw[1],
#                    int(bbox_raw[2] - raw_difference),
#                    bbox_raw[3]
#                  ]
#                  new_bboxes_raw.append([max(0, b) for b in new_bbox_raw])
#             col_bboxes_raw = new_bboxes_raw
        
#         print(f"  (Page {page_num + 1}, Column {col_idx + 1}) Horizontal shift applied.")      
            

def run_inference_and_structure(pdf_path: str, model_path: str, inference_output_path: str, 
                                column_detection_params: Dict = None) -> List[Dict[str, Any]]:
    labels = [
        "O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION",
        "B-ANSWER", "I-ANSWER", "B-SECTION_HEADING", "I-SECTION_HEADING",
        "B-PASSAGE", "I-PASSAGE"
    ]

    if column_detection_params is None:
        column_detection_params = {
            'edge_margin_percent': 0.05,
            'merge_tolerance': 30,
            'top_margin_percent': 0.10,
            'bottom_margin_percent': 0.10,
            'density_resolution': 2,
            'density_smoothing': 3,
            'gutter_prominence': 0.15,
            'gutter_min_width': 5,
            'gutter_min_distance': 50,
            'gutter_relative_prominence': 0.50,
            'cluster_bin_size': 5,
            'cluster_smoothing': 2,
            'cluster_threshold_percentile': 20,
            'cluster_min_width': 20,
        }

    id2label = {i: l for i, l in enumerate(labels)}
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3CRF("microsoft/layoutlmv3-base", num_labels=len(labels)).to(device)

    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at {model_path}. Please check the path.")
        return []

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully from {model_path} with {len(labels)} labels.")

    try:
        doc = pdfplumber.open(pdf_path)
    except Exception as e:
        print(f"❌ Error opening PDF with pdfplumber: {e}")
        return []

    all_pages_data = []
    CHUNK_SIZE = 500

    for page_num, page in enumerate(doc.pages):
        print(f"\nProcessing page {page_num + 1}...")

        page_width, page_height = page.width, page.height
        words = []
        bboxes_raw = []
        normalized_bboxes = []

        # --- Word Extraction (PDFPlumber) ---
        word_list = page.extract_words(x_tolerance=1, y_tolerance=1, keep_blank_chars=False, use_text_flow=True)
        if word_list:
            print(f"  (Page {page_num + 1}) Found {len(word_list)} words using pdfplumber.")
            for word_data in word_list:
                word = word_data['text']
                raw_bbox = [word_data['x0'], word_data['top'], word_data['x1'], word_data['bottom']]

                words.append(word)
                bboxes_raw.append([int(b) for b in raw_bbox])

                normalized_bboxes.append([
                    max(0, min(1000, int(1000 * raw_bbox[0] / page_width))),
                    max(0, min(1000, int(1000 * raw_bbox[1] / page_height))),
                    max(0, min(1000, int(1000 * raw_bbox[2] / page_width))),
                    max(0, min(1000, int(1000 * raw_bbox[3] / page_height)))
                ])

        # --- OCR Fallback ---
        if not words:
            print(f"  (Page {page_num + 1}) No text layer found. Running Tesseract OCR fallback...")
            try:
                fitz_doc = fitz.open(pdf_path)
                fitz_page = fitz_doc.load_page(page_num)
                pix = fitz_page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                fitz_doc.close()
                words, bboxes_raw, normalized_bboxes = run_tesseract_ocr(img, page_width, page_height)
            except Exception as e:
                print(f"  (Page {page_num + 1}) Error during OCR fallback: {e}")

            if not words:
                print(f"  (Page {page_num + 1}) OCR failed to find any words. Skipping page.")
                continue

        # --- NEW COLUMN DETECTION INTEGRATION ---
        separator_x = detect_column_gutters(pdf_path, page_num, **column_detection_params)
        
        # Force conversion to int if it's a list (safety check)
        if separator_x is not None:
            if isinstance(separator_x, list):
                separator_x = int(separator_x[0]) if separator_x else None
        
        if separator_x is not None:
            print(f"  (Page {page_num + 1}) ✅ Column separator detected at X={separator_x}")
            split_data = split_data_by_gutter(words, bboxes_raw, normalized_bboxes, 
                                              separator_x, page_width, page_height)
        else:
            print(f"  (Page {page_num + 1}) Single column structure assumed.")
            split_data = []

        # If no split occurred, treat as single column
        if not split_data:
            split_data.append({
                "words": words,
                "bboxes_raw": bboxes_raw,
                "normalized_bboxes": normalized_bboxes
            })

        # Apply preprocessing to EACH column (whether split or not)
        for col_idx, col_data in enumerate(split_data):
            print(f"  DEBUG: Preprocessing column {col_idx + 1}, has {len(col_data['words'])} words")
            col_words = col_data['words']
            col_bboxes_raw = col_data['bboxes_raw']
            col_normalized_bboxes = col_data['normalized_bboxes']

            if col_normalized_bboxes:
                min_x_min = min(bbox[0] for bbox in col_normalized_bboxes)
                if min_x_min < 110:
                    difference = min_x_min - 110
                
                    # Adjust normalized bboxes
                    new_normalized_bboxes = []
                    for bbox in col_normalized_bboxes:
                        new_bbox = [
                            bbox[0] - difference, bbox[1],
                            bbox[2] - difference, bbox[3]
                        ]
                        new_normalized_bboxes.append([min(1000, max(0, b)) for b in new_bbox])
                    col_normalized_bboxes = new_normalized_bboxes
                    
                    # ALSO adjust raw bboxes proportionally
                    raw_difference = difference * page_width / 1000.0  # Convert normalized difference to raw pixels
                    new_bboxes_raw = []
                    for bbox_raw in col_bboxes_raw:
                        new_bbox_raw = [
                            int(bbox_raw[0] - raw_difference),
                            bbox_raw[1],
                            int(bbox_raw[2] - raw_difference),
                            bbox_raw[3]
                        ]
                        new_bboxes_raw.append([max(0, b) for b in new_bbox_raw])
                    col_bboxes_raw = new_bboxes_raw
            
            print(f"  (Page {page_num + 1}, Column {col_idx + 1}) Horizontal shift applied.")            
            
            # Pre-processing 1: Horizontal shift adjustment
            
            # if col_normalized_bboxes:
            #     min_x_min = min(bbox[0] for bbox in col_normalized_bboxes)
            #     if min_x_min < 110:
            #         difference = min_x_min - 110
            #         new_normalized_bboxes = []
            #         for bbox in col_normalized_bboxes:
            #             new_bbox = [
            #                 bbox[0] - difference, bbox[1],
            #                 bbox[2] - difference, bbox[3]
            #             ]
            #             new_normalized_bboxes.append([min(1000, max(0, b)) for b in new_bbox])
            #         col_normalized_bboxes = new_normalized_bboxes
            #         print(f"  (Page {page_num + 1}, Column {col_idx + 1}) Horizontal shift applied.")
            
            # Pre-processing 2: Rightmost numerical token removal
            if col_normalized_bboxes:
                max_x_max_val = max(bbox[2] for bbox in col_normalized_bboxes)
                rightmost_indices = [i for i, bbox in enumerate(col_normalized_bboxes) if bbox[2] >= max_x_max_val - 1]
                removed_count = 0
                
                temp_words = col_words[:]
                temp_bboxes_raw = col_bboxes_raw[:]
                temp_normalized_bboxes = col_normalized_bboxes[:]
                
                for i in reversed(rightmost_indices):
                    word = temp_words[i].strip()
                    is_number = word.isdigit() or \
                                (word.replace('.', '', 1).isdigit() if '.' in word else False) or \
                                (word.replace(',', '').isdigit() if ',' in word else False) or \
                                (word.startswith('(') and word.endswith(')') and word[1:-1].isdigit())
                    
                    if is_number:
                        temp_words.pop(i)
                        temp_bboxes_raw.pop(i)
                        temp_normalized_bboxes.pop(i)
                        removed_count += 1
                
                col_words = temp_words
                col_bboxes_raw = temp_bboxes_raw
                col_normalized_bboxes = temp_normalized_bboxes
                
                if removed_count > 0:
                    print(f"  (Page {page_num + 1}, Column {col_idx + 1}) Removed {removed_count} rightmost numerical tokens.")
            
            # Update the column data with preprocessed values
            split_data[col_idx] = {
                "words": col_words,
                "bboxes_raw": col_bboxes_raw,
                "normalized_bboxes": col_normalized_bboxes
            }

        # --- ITERATE OVER COLUMNS FOR INFERENCE ---
        for col_idx, col_data in enumerate(split_data):
            words = col_data['words']
            bboxes_raw = col_data['bboxes_raw']
            normalized_bboxes = col_data['normalized_bboxes']

            if not words:
                continue

            page_final_preds = []
            word_idx_start = 0

            while word_idx_start < len(words):
                current_words = words[word_idx_start:]
                current_bboxes = normalized_bboxes[word_idx_start:]

                tokenized_chunk = tokenizer(current_words, boxes=current_bboxes, return_offsets_mapping=True)
                split_token_idx = min(len(tokenized_chunk.input_ids), CHUNK_SIZE)
                word_ids = tokenized_chunk.word_ids()
                split_word_idx = -1

                for token_idx in range(split_token_idx - 1, -1, -1):
                    word_id = word_ids[token_idx]
                    if word_id is not None:
                        split_word_idx = word_id + 1
                        break

                if split_word_idx <= 0:
                    split_word_idx = 1

                chunk_words = current_words[:split_word_idx]
                chunk_bboxes = current_bboxes[:split_word_idx]

                inputs = tokenizer(chunk_words, boxes=chunk_bboxes, return_tensors="pt", truncation=True,
                                   max_length=CHUNK_SIZE, padding=False).to(device)

                if inputs.input_ids.shape[1] <= 2:
                    break

                with torch.no_grad():
                    preds = model(**inputs)

                chunk_word_ids = inputs.word_ids(batch_index=0)
                chunk_preds = []
                word_to_first_token = {}
                for token_idx, word_id in enumerate(chunk_word_ids):
                    if word_id is not None and word_id not in word_to_first_token:
                        word_to_first_token[word_id] = token_idx

                for word_id in range(len(chunk_words)):
                    if word_id in word_to_first_token:
                        token_idx = word_to_first_token[word_id]
                        if token_idx < len(preds[0]):
                            chunk_preds.append(id2label[preds[0][token_idx]])
                        else:
                            chunk_preds.append("O")
                    else:
                        chunk_preds.append("O")

                page_final_preds.extend(chunk_preds)
                word_idx_start += split_word_idx

            if len(words) == len(page_final_preds):
                page_results = []
                for word, bbox_raw, label in zip(words, bboxes_raw, page_final_preds):
                    page_results.append({
                        "word": word,
                        "bbox": bbox_raw,
                        "predicted_label": label
                    })

                all_pages_data.append({
                    "page_num": page_num,
                    "column_index": col_idx,
                    "width": page_width,
                    "height": page_height,
                    "original_words": words,
                    "original_bboxes": bboxes_raw,
                    "results": page_results
                })
                print(f"✅ Page {page_num + 1} (Column {col_idx + 1}) processed successfully.")
            else:
                print(f"❌ [FATAL Error] Final prediction/word mismatch on page {page_num + 1} (Column {col_idx + 1}). Skipping column results.")

    doc.close()

    with open(inference_output_path, "w", encoding='utf-8') as f:
        json.dump(all_pages_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Structured predictions saved to {inference_output_path}")

    return all_pages_data


def convert_to_label_studio_format(structured_data: List[Dict[str, Any]], output_path: str):
    final_tasks = []

    for page_data in structured_data:
        page_num = page_data['page_num']
        col_idx = page_data['column_index']
        text_string = " ".join(page_data['original_words'])

        results = []
        current_entity_label: Optional[str] = None
        current_entity_start_word_index: Optional[int] = None

        for i, pred_item in enumerate(page_data['results']):
            label = pred_item['predicted_label']

            if label.startswith('B-'):
                # If an entity was open, close it
                if current_entity_label:
                    results.append(create_label_studio_span(
                        page_data['results'],
                        current_entity_start_word_index,
                        i - 1,
                        current_entity_label
                    ))
                current_entity_label = label[2:]
                current_entity_start_word_index = i

            elif label.startswith('I-') and current_entity_label == label[2:]:
                # Continue the same entity
                continue
            else:
                # If label is 'O' or label does not match current entity
                if current_entity_label:
                    results.append(create_label_studio_span(
                        page_data['results'],
                        current_entity_start_word_index,
                        i - 1,
                        current_entity_label
                    ))
                    current_entity_label = None
                    current_entity_start_word_index = None

        # Flush last entity if still open
        if current_entity_label:
            results.append(create_label_studio_span(
                page_data['results'],
                current_entity_start_word_index,
                len(page_data['results']) - 1,
                current_entity_label
            ))

        task = {
            "data": {
                "text": text_string,
                "original_words": page_data['original_words'],
                "original_bboxes": page_data['original_bboxes']
            },
            "annotations": [
                {
                    "result": results
                }
            ],
            "meta": {
                "page_number": page_num + 1,
                "column_index": col_idx + 1
            }
        }
        final_tasks.append(task)

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_tasks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Label Studio tasks created in OCR format and saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LayoutLMv3 Inference Pipeline for PDF and Label Studio OCR Conversion.")
    parser.add_argument("--input_pdf", type=str, required=True,
                        help="Path to the input PDF file for inference.")
    parser.add_argument("--model_path", type=str, default="checkpoints/layoutlmv3_trained_20251031_102846_recovered.pth",
                        help="Path to the saved LayoutLMv3-CRF PyTorch model checkpoint.")
    parser.add_argument("--inference_output", type=str, default="structured_predictions.json",
                        help="Path to save the intermediate structured predictions.")
    parser.add_argument("--label_studio_output", type=str, default="label_studio_import.json",
                        help="Path to save the final Label Studio import JSON.")
    parser.add_argument("--no_labelstudio", action="store_true",
                        help="If set, skip creating the Label Studio import JSON and only write structured predictions.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose printing.")

    try:
        import fitz  # keep the optional import check you had
    except ImportError:
        print("\n⚠️ WARNING: PyMuPDF ('fitz') is required for OCR fallback. Install with 'pip install PyMuPDF'.")

    args = parser.parse_args()

    # ensure model folder exists (same as in your original)
    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)

    # Column detection parameters (keep your tuned values)
    column_params = {
        'edge_margin_percent': 0.05,
        'merge_tolerance': 30,
        'top_margin_percent': 0.10,
        'bottom_margin_percent': 0.10,
        'density_resolution': 2,
        'density_smoothing': 3,
        'gutter_prominence': 0.60,
        'gutter_min_width': 5,
        'gutter_min_distance': 70,
        'gutter_relative_prominence': 0.70,
        'cluster_bin_size': 5,
        'cluster_smoothing': 2,
        'cluster_threshold_percentile': 30,
        'cluster_min_width': 25,
    }

    # Run inference -> produces structured predictions and writes inference_output
    try:
        structured_data = run_inference_and_structure(
            args.input_pdf,
            args.model_path,
            args.inference_output,
            column_detection_params=column_params
        )
    except Exception as e:
        print(f"❌ Fatal error while running inference: {e}")
        structured_data = []

    # If requested, convert to Label Studio format
    if structured_data and not args.no_labelstudio:
        try:
            convert_to_label_studio_format(
                structured_data=structured_data,
                output_path=args.label_studio_output
            )
        except Exception as e:
            print(f"❌ Error while converting to Label Studio format: {e}")
    elif not structured_data:
        print("⚠️ No structured data produced — skipping Label Studio conversion.")
    else:
        print("ℹ️ Skipped Label Studio conversion as requested (--no_labelstudio).")

    # final status message
    if args.verbose:
        print(f"\nFinished. Structured predictions file: {os.path.abspath(args.inference_output)}")
        if os.path.exists(args.label_studio_output):
            print(f"Label Studio import file: {os.path.abspath(args.label_studio_output)}")