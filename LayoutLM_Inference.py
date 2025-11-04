#Not Required File
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


# --- Model Architecture  ---
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


# --- Helper function for OCR fallback (Needs PIL image input) ---
def run_tesseract_ocr(img: Image, page_width, page_height) -> tuple[list, list]:
    """Helper to run Tesseract and normalize bounding boxes."""
    tesseract_config = '--psm 6'

    # 1. Get OCR data
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=tesseract_config)
    valid_indices = [i for i, word in enumerate(ocr_data['text']) if word.strip()]

    words = [ocr_data['text'][i] for i in valid_indices]

    # Calculate scale factors based on the assumption that Tesseract image was 300 DPI
    x_scale = page_width / img.width
    y_scale = page_height / img.height

    bboxes_raw = []
    for i in valid_indices:
        # Tesseract returns pixel coordinates (left, top, width, height)
        left = ocr_data['left'][i]
        top = ocr_data['top'][i]
        width = ocr_data['width'][i]
        height = ocr_data['height'][i]

        # Convert Tesseract pixels back to PDF coordinates (raw_bbox)
        raw_bbox = [
            left * x_scale,
            top * y_scale,
            (left + width) * x_scale,
            (top + height) * y_scale
        ]
        bboxes_raw.append([int(b) for b in raw_bbox])

    # 2. Normalize bboxes
    normalized_bboxes = [[
        int(1000 * b[0] / page_width),
        int(1000 * b[1] / page_height),
        int(1000 * b[2] / page_width),
        int(1000 * b[3] / page_height)
    ] for b in bboxes_raw]

    return words, bboxes_raw, normalized_bboxes


# -----------------------------------------------------------
# run_inference FUNCTION USING PDFPLUMBER
# -----------------------------------------------------------
def run_inference(pdf_path: str, model_path: str, output_path: str):
    labels = ["O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION", "B-ANSWER", "I-ANSWER"]
    id2label = {i: l for i, l in enumerate(labels)}

    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3CRF("microsoft/layoutlmv3-base", num_labels=len(labels)).to(device)

    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at {model_path}. Please check the path.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully from {model_path}.")

    try:
        # Open PDF using pdfplumber
        doc = pdfplumber.open(pdf_path)
    except Exception as e:
        print(f"❌ Error opening PDF with pdfplumber: {e}")
        return

    all_predictions = []
    CHUNK_SIZE = 500

    for page_num, page in enumerate(doc.pages):
        print(f"Processing page {page_num + 1}...")

        page_width, page_height = page.width, page.height
        words = []
        bboxes_raw = []
        normalized_bboxes = []

        # -----------------------------------------------------------
        # PRIMARY EXTRACTION: PDFPLUMBER TEXT LAYER
        # -----------------------------------------------------------
        word_list = page.extract_words(x_tolerance=1, y_tolerance=1, keep_blank_chars=False, use_text_flow=True)

        if word_list:
            print(f"  (Page {page_num + 1}) Found {len(word_list)} words using pdfplumber. Skipping OCR.")
            for word_data in word_list:
                word = word_data['text']

                # pdfplumber bounding box keys: 'x0', 'top', 'x1', 'bottom'
                raw_bbox = [word_data['x0'], word_data['top'], word_data['x1'], word_data['bottom']]

                words.append(word)
                bboxes_raw.append([int(b) for b in raw_bbox])

                # Normalize bbox
                normalized_bboxes.append([
                    int(1000 * raw_bbox[0] / page_width),
                    int(1000 * raw_bbox[1] / page_height),
                    int(1000 * raw_bbox[2] / page_width),
                    int(1000 * raw_bbox[3] / page_height)
                ])

        # -----------------------------------------------------------
        # FALLBACK: TESSERACT OCR
        # pdfplumber does not have a native method to render images at high DPI,
        # so we will use the standard PyMuPDF/PIL approach for the fallback.
        # This requires re-opening the PDF with fitz just for the image rendering.
        # -----------------------------------------------------------
        if not words:
            print(f"  (Page {page_num + 1}) No text layer found. Running Tesseract OCR fallback...")

            # Temporary re-open using fitz for image rendering
            try:
                import fitz
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

        if not words:
            continue

        # --- CHUNKING LOGIC---
        page_final_preds = []
        word_idx_start = 0

        while word_idx_start < len(words):
            current_words = words[word_idx_start:]
            current_bboxes = normalized_bboxes[word_idx_start:]

            # 1. First tokenization to determine split point
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

            # 2. Second tokenization for actual inference
            inputs = tokenizer(chunk_words, boxes=chunk_bboxes, return_tensors="pt", truncation=True,
                               max_length=CHUNK_SIZE, padding=False).to(device)

            if inputs.input_ids.shape[1] <= 2:
                break

            with torch.no_grad():
                preds = model(**inputs)

            # 3. Robust alignment: map predictions to words
            chunk_word_ids = inputs.word_ids(batch_index=0)
            chunk_preds = []

            word_to_first_token = {}
            for token_idx, word_id in enumerate(chunk_word_ids):
                if word_id is not None and word_id not in word_to_first_token:
                    word_to_first_token[word_id] = token_idx

            for word_id in range(len(chunk_words)):
                if word_id in word_to_first_token:
                    token_idx = word_to_first_token[word_id]
                    chunk_preds.append(id2label[preds[0][token_idx]])
                else:
                    chunk_preds.append("O")

            if len(chunk_words) == len(chunk_preds):
                page_final_preds.extend(chunk_preds)
            else:
                print(
                    f"❌ [CRITICAL Mismatch] on page {page_num + 1} chunk: {len(chunk_words)} words vs {len(chunk_preds)} preds. Advancing index despite mismatch.")
                page_final_preds.extend(chunk_preds)

            word_idx_start += split_word_idx

        # --- END OF CHUNKING LOGIC ---

        # Final aggregation check
        if len(words) == len(page_final_preds):
            page_results = []
            for word, bbox, label in zip(words, bboxes_raw, page_final_preds):
                page_results.append({
                    "word": word,
                    "bbox": bbox,
                    "predicted_label": label
                })
            all_predictions.extend(page_results)
            print(f"✅ Page {page_num + 1} processed successfully with {len(words)} words in total.")
        else:
            print(
                f"❌ [FATAL Error] Final prediction/word mismatch on page {page_num + 1}: {len(words)} words vs {len(page_final_preds)} total preds. Skipping page results.")

    doc.close()
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Inference complete. Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LayoutLMv3 Inference Script for PDF to BIO-JSON conversion.")
    parser.add_argument("--input_pdf", type=str, required=True,
                        help="Path to the input PDF file for inference.")
    parser.add_argument("--model_path", type=str, default="checkpoints/layoutlmv3_crf_new.pth",
                        help="Path to the saved LayoutLMv3-CRF PyTorch model checkpoint.")
    parser.add_argument("--output_json", type=str, default="inference_predictions2.json",
                        help="Path to save the output BIO-encoded JSON predictions.")

    args = parser.parse_args()

    run_inference(args.input_pdf, args.model_path, args.output_json)
