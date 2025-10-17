
import json
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Model
from TorchCRF import CRF
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# --- Configuration for Augmentation ---
MAX_BBOX_DIMENSION = 999
MAX_SHIFT = 30
AUGMENTATION_FACTOR = 1


# -------------------------------------


# -------------------------
# Step 1: Preprocessing (Label Studio → BIO + bboxes)
# -------------------------
def preprocess_labelstudio(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    for item in data:
        words = item["data"]["original_words"]
        bboxes = item["data"]["original_bboxes"]
        labels = ["O"] * len(words)

        if "annotations" in item:
            for ann in item["annotations"]:
                for res in ann["result"]:
                    # Check if the result item is a span annotation
                    if "value" in res and "labels" in res["value"]:
                        text = res["value"]["text"]
                        tag = res["value"]["labels"][0]
                        # Some tokenizers may split words, so we must find a consecutive word match.
                        text_tokens = text.split()

                        for i in range(len(words) - len(text_tokens) + 1):
                            if words[i:i + len(text_tokens)] == text_tokens:
                                labels[i] = f"B-{tag}"
                                for j in range(1, len(text_tokens)):
                                    labels[i + j] = f"I-{tag}"
                                break  # Move to next annotation if a match is found

        processed.append({"tokens": words, "labels": labels, "bboxes": bboxes})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"✅ Preprocessed data saved to {output_path}")
    return output_path


# -------------------------
# Step 1.5: Bounding Box Augmentation
# -------------------------

def translate_bbox(bbox, shift_x, shift_y):
    """
    Translates a single bounding box [x_min, y_min, x_max, y_max] by (shift_x, shift_y)
    and clamps the coordinates to the valid range [0, MAX_BBOX_DIMENSION].
    """
    x_min, y_min, x_max, y_max = bbox

    new_x_min = x_min + shift_x
    new_y_min = y_min + shift_y
    new_x_max = x_max + shift_x
    new_y_max = y_max + shift_y

    # Clamp the new coordinates
    new_x_min = max(0, min(new_x_min, MAX_BBOX_DIMENSION))
    new_y_min = max(0, min(new_y_min, MAX_BBOX_DIMENSION))
    new_x_max = max(0, min(new_x_max, MAX_BBOX_DIMENSION))
    new_y_max = max(0, min(new_y_max, MAX_BBOX_DIMENSION))

    # Safety check
    if new_x_min > new_x_max: new_x_min = new_x_max
    if new_y_min > new_y_max: new_y_min = new_y_max

    return [new_x_min, new_y_min, new_x_max, new_y_max]


def augment_sample(sample):
    """
    Generates a new sample by translating all bounding boxes.
    """
    shift_x = random.randint(-MAX_SHIFT, MAX_SHIFT)
    shift_y = random.randint(-MAX_SHIFT, MAX_SHIFT)

    new_sample = sample.copy()

    # Ensure tokens and labels are copied (they remain unchanged)
    new_sample["tokens"] = sample["tokens"]
    new_sample["labels"] = sample["labels"]

    # Translate all bounding boxes
    new_bboxes = [translate_bbox(bbox, shift_x, shift_y) for bbox in sample["bboxes"]]
    new_sample["bboxes"] = new_bboxes

    return new_sample


def augment_and_save_dataset(input_json_path, output_json_path):
    """
    Loads preprocessed data, performs augmentation, and saves the result.
    """
    print(f"🔄 Loading preprocessed data from {input_json_path} for augmentation...")
    with open(input_json_path, 'r', encoding="utf-8") as f:
        training_data = json.load(f)

    augmented_data = []
    original_count = len(training_data)

    for i, original_sample in enumerate(training_data):
        # 1. Add the original sample
        augmented_data.append(original_sample)

        # 2. Generate augmented samples
        for _ in range(AUGMENTATION_FACTOR):
            if "tokens" in original_sample and "labels" in original_sample and "bboxes" in original_sample:
                augmented_data.append(augment_sample(original_sample))
            else:
                print(f"Warning: Skipping augmentation for sample {i} due to missing keys.")

    augmented_count = len(augmented_data)
    print(f"Dataset Augmentation: Original samples: {original_count}, Total samples: {augmented_count}")

    # Save the augmented dataset
    with open(output_json_path, 'w', encoding="utf-8") as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Augmented data saved to {output_json_path}")
    return output_json_path


# -------------------------
# Step 2: Dataset Class (Unchanged)
# -------------------------
class LayoutDataset(Dataset):
    def __init__(self, json_path, tokenizer, label2id, max_len=512):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words, bboxes, labels = item["tokens"], item["bboxes"], item["labels"]

        # Tokenize
        encodings = self.tokenizer(
            words,
            boxes=bboxes,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Align labels to word pieces
        word_ids = encodings.word_ids(batch_index=0)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(self.label2id["O"])  # [CLS], [SEP], padding
            else:
                label_ids.append(self.label2id.get(labels[word_id], self.label2id["O"]))

        encodings.pop("offset_mapping")
        encodings["labels"] = torch.tensor(label_ids)

        return {key: val.squeeze(0) for key, val in encodings.items()}


# -------------------------
# Step 3: Model Architecture 
# -------------------------
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
            # Training mode: calculate loss
            log_likelihood = self.crf(emissions, labels, mask=attention_mask.bool())
            return -log_likelihood.mean()
        else:
            # Inference mode: decode best path
            best_paths = self.crf.viterbi_decode(emissions, mask=attention_mask.bool())
            return best_paths


# -------------------------
# Step 4: Training + Evaluation 
# -------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        optimizer.zero_grad()
        loss = model(**batch, labels=labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, id2label):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels").cpu().numpy()
            preds = model(**batch)
            for p, l, mask in zip(preds, labels, batch["attention_mask"].cpu().numpy()):
                valid = mask == 1
                l = l[valid].tolist()
                all_labels.extend(l)
                all_preds.extend(p[:len(l)])

    # Exclude the "O" label and other special tokens if necessary, but using 'micro' average
    # on all valid tokens is typically fine for the initial evaluation.
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro", zero_division=0)
    return precision, recall, f1


# -------------------------
# Step 5: Main Pipeline (Training) 
# -------------------------
def main(args):
    # Labels (extend as needed)
    labels = ["O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION", "B-ANSWER", "I-ANSWER"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # 1. Preprocess and save the initial training data
    initial_bio_json = "training_data_bio_bboxes.json"
    preprocess_labelstudio(args.input, initial_bio_json)

    # 2. Augment the dataset with translated bboxes
    augmented_bio_json = "augmented_training_data_bio_bboxes.json"
    final_data_path = augment_and_save_dataset(initial_bio_json, augmented_bio_json)

    # Clean up the intermediary file (optional)
    # os.remove(initial_bio_json)

    # 3. Load and split augmented dataset
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    dataset = LayoutDataset(final_data_path, tokenizer, label2id, max_len=args.max_len)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size

    # Use a fixed seed for reproducibility in split
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 4. Initialize and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3CRF("microsoft/layoutlmv3-base", num_labels=len(labels)).to(device)
    ckpt_path = "checkpoints/layoutlmv3_crf_new.pth"
    os.makedirs("checkpoints", exist_ok=True)
    if os.path.exists(ckpt_path):
        print(f"🔄 Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 5. Training loop
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        precision, recall, f1 = evaluate(model, val_loader, device, id2label)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Model saved at {ckpt_path}")


def run_inference(pdf_path, model_path, output_path):
    # Load labels and tokenizer
    labels = ["O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION", "B-ANSWER", "I-ANSWER"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3CRF("microsoft/layoutlmv3-base", num_labels=len(labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Process PDF with OCR
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return

    all_predictions = []
    tesseract_config = '--psm 6'

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Get a high-resolution image of the page for Tesseract
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Get page dimensions from PyMuPDF
        page_width, page_height = page.bound().width, page.bound().height

        # Get OCR data (words and bboxes)
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=tesseract_config)
        words = [word for word in ocr_data['text'] if word.strip()]

        # Skip empty pages
        if not words:
            continue

        # Get the scaling factors from the image resolution to the PDF's native resolution
        x_scale = page_width / pix.width
        y_scale = page_height / pix.height

        # Create original pixel bboxes
        bboxes_raw = [[
            ocr_data['left'][i],
            ocr_data['top'][i],
            ocr_data['left'][i] + ocr_data['width'][i],
            ocr_data['top'][i] + ocr_data['height'][i]
        ] for i in range(len(ocr_data['text'])) if ocr_data['text'][i].strip()]

        # Normalize bboxes to 0-1000 scale using the correct scaling factors
        normalized_bboxes = [[
            int(1000 * (b[0] * x_scale) / page_width),
            int(1000 * (b[1] * y_scale) / page_height),
            int(1000 * (b[2] * x_scale) / page_width),
            int(1000 * (b[3] * y_scale) / page_height)
        ] for b in bboxes_raw]

        # Tokenize and run inference
        inputs = tokenizer(words, boxes=normalized_bboxes, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            # The model is run on the normalized bboxes
            preds = model(**inputs)

        # Align predictions back to words
        word_ids = inputs.word_ids(batch_index=0)
        final_preds = []
        previous_word_idx = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous_word_idx:
                # The model returns a list of predicted classes for each token
                final_preds.append(id2label[preds[0][idx]])
            previous_word_idx = word_id

        # Prepare structured output
        page_results = []
        for word, bbox, label in zip(words, bboxes_raw, final_preds):
            page_results.append({
                "word": word,
                "bbox": bbox,
                "predicted_label": label
            })
        all_predictions.extend(page_results)

    doc.close()
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    print(f"✅ Inference complete. Predictions saved to {output_path}")


# -------------------------
# Step 7: Main Execution 
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LayoutLMv3 Fine-tuning and Inference Script.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "infer"],
                        help="Select mode: 'train' or 'infer'")
    parser.add_argument("--input", type=str, help="Path to input file (Label Studio JSON for train, PDF for infer).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()

    if args.mode == "train":
        if not args.input:
            parser.error("--input is required for 'train' mode.")
        main(args)
    elif args.mode == "infer":
        if not args.input:
            parser.error("--input is required for 'infer' mode.")
        run_inference(args.input, "checkpoints/layoutlmv3_crf_new.pth", "inference_predictions.json")
