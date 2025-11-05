#Not Required File

import json
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# Using LayoutLMv3TokenizerFast, LayoutLMv3Model
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Model
from transformers.utils import cached_file
from safetensors.torch import load_file
from TorchCRF import CRF
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# --- Configuration for Augmentation ---
MAX_BBOX_DIMENSION = 1000  # Corrected to 1000 to match LayoutLMv3 requirement
MAX_SHIFT = 30
AUGMENTATION_FACTOR = 1

# -------------------------------------

# --- Hugging Face Model ID ---
HF_MODEL_ID = "heerjtdev/edugenius"


# -----------------------------


# -------------------------
# Step 1: Preprocessing (Label Studio → BIO + bboxes)
# -------------------------
def preprocess_labelstudio(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = []
    total_items = len(data)  # Added for potential verbose logging
    print(f"🔄 Starting preprocessing of {total_items} documents My name is Aastik!! BOOBS...")

    for item in data:
        words = item["data"]["original_words"]
        bboxes = item["data"]["original_bboxes"]
        labels = ["O"] * len(words)

        # --- NEW: Bounding Box Normalization/Clamping ---
        # Defensively ensures all coordinates are within the [0, 1000] range
        # required by LayoutLMv3's spatial position embeddings.
        clamped_bboxes = []
        for bbox in bboxes:
            # Clamp coordinates to [0, 1000]
            x_min, y_min, x_max, y_max = bbox

            new_x_min = max(0, min(x_min, 1000))
            new_y_min = max(0, min(y_min, 1000))
            new_x_max = max(0, min(x_max, 1000))
            new_y_max = max(0, min(y_max, 1000))

            # Safety check: ensure min <= max (this should rarely trigger
            # if the original bboxes were valid, but is good practice)
            if new_x_min > new_x_max: new_x_min = new_x_max
            if new_y_min > new_y_max: new_y_min = new_y_max

            clamped_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])

        # Use the clamped bboxes for the rest of the pipeline
        final_bboxes = clamped_bboxes
        # ------------------------------------------------

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

        processed.append({"tokens": words, "labels": labels, "bboxes": final_bboxes})

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

    # Clamp the new coordinates (MAX_BBOX_DIMENSION is 1000)
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

    print(f"🔄 Starting augmentation (Factor: {AUGMENTATION_FACTOR}, {original_count} documents)...")

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
# Step 2: Dataset Class
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
# Step 3: Model Architecture (PATCHED TO LOAD WEIGHTS CORRECTLY)
# -------------------------
class LayoutLMv3CRF(nn.Module):
    def __init__(self, model_name, num_labels, device):
        super().__init__()

        # 1. Initialize the LayoutLMv3 model using the base class
        # We start by initializing from the base configuration to ensure all weights are present
        self.layoutlm = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")

        # 2. Try to load the fine-tuned weights from the Hugging Face Hub/Cache
        try:
            # This resolves the path to the downloaded model.safetensors in the cache
            # Assumes you have renamed your file on the Hugging Face Hub to 'model.safetensors'
            weights_path = cached_file(model_name, "model.safetensors")
            fine_tuned_weights = load_file(weights_path)

            # 3. Strip the Mismatching Prefix (Assuming 'layoutlm.' prefix from a previous wrapper)
            new_state_dict = {}
            prefix_to_strip = "layoutlm."

            for key, value in fine_tuned_weights.items():
                if key.startswith(prefix_to_strip):
                    new_key = key[len(prefix_to_strip):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            # 4. Load the fixed state dictionary into the LayoutLMv3Model
            # strict=False allows us to ignore classifier/CRF weights not in LayoutLMv3Model
            print("🔄 Successfully loaded and stripped keys. Loading base LayoutLMv3 weights...")

            # Load only the weights for the transformer body
            missing_keys, unexpected_keys = self.layoutlm.load_state_dict(new_state_dict, strict=False)

            print(f"Weights loading done: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys.")

        except Exception as e:
            print(f"❌ Fine-tuned weights could not be loaded directly and mapped. Starting with random weights.")
            print(f"Error: {e}")
            # Fallback: Load the LayoutLMv3 component directly from the Hub ID (will result in random weights for layers)
            self.layoutlm = LayoutLMv3Model.from_pretrained(model_name)

        # 5. Initialize the new heads (CRF layer and Classifier)
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
            # The model returns a list of lists of predicted labels in inference mode
            preds = model(**batch)
            for p, l, mask in zip(preds, labels, batch["attention_mask"].cpu().numpy()):
                valid = mask == 1
                l = l[valid].tolist()
                all_labels.extend(l)
                # Ensure pred length matches label length for the unmasked tokens
                all_preds.extend(p[:len(l)])

                # Exclude the "O" label and other special tokens if necessary, but using 'micro' average
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro", zero_division=0)
    return precision, recall, f1


# -------------------------
# Step 5: Main Pipeline (Training) - MODIFIED MODEL/TOKENIZER LOADING
# -------------------------
def main(args):
    # LABELS UPDATED: Added SECTION_HEADING and PASSAGE
    labels = [
        "O",
        "B-QUESTION", "I-QUESTION",
        "B-OPTION", "I-OPTION",
        "B-ANSWER", "I-ANSWER",
        "B-SECTION_HEADING", "I-SECTION_HEADING",
        "B-PASSAGE", "I-PASSAGE"
    ]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    # --- SETUP: Use a temporary directory for intermediate files ---
    TEMP_DIR = "temp_intermediate_files"
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"\n--- SETUP PHASE: Created temp directory: {TEMP_DIR} ---")

    # 1. Preprocess
    print("\n--- START PHASE: PREPROCESSING ---")
    initial_bio_json = os.path.join(TEMP_DIR, "training_data_bio_bboxes.json")
    preprocess_labelstudio(args.input, initial_bio_json)

    # 2. Augment
    print("\n--- START PHASE: AUGMENTATION ---")
    augmented_bio_json = os.path.join(TEMP_DIR, "augmented_training_data_bio_bboxes.json")
    final_data_path = augment_and_save_dataset(initial_bio_json, augmented_bio_json)

    # 3. Load and split augmented dataset
    print("\n--- START PHASE: MODEL/DATASET SETUP ---")

    # Load tokenizer from the specified Hugging Face ID
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(HF_MODEL_ID)

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
    print(f"Using device: {device}")

    # Pass the Hugging Face ID and device to the custom model wrapper
    model = LayoutLMv3CRF(HF_MODEL_ID, num_labels=len(labels), device=device).to(device)

    ckpt_path = "checkpoints/layoutlmv3_crf_passage.pth"
    os.makedirs("checkpoints", exist_ok=True)
    if os.path.exists(ckpt_path):
        print(f"⚠️ Starting fresh training. Old checkpoint {ckpt_path} may be incompatible with new label count.")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 5. Training loop
    for epoch in range(args.epochs):
        print(f"\n--- START PHASE: EPOCH {epoch + 1}/{args.epochs} TRAINING ---")
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)

        print(f"\n--- START PHASE: EPOCH {epoch + 1}/{args.epochs} EVALUATION ---")
        precision, recall, f1 = evaluate(model, val_loader, device, id2label)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Model saved at {ckpt_path}")


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