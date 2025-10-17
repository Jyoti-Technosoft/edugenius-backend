
import os
import re
import json
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure TorchCRF is installed: pip install torch-crf
try:
    from TorchCRF import CRF
except ImportError:
    print("Error: The 'TorchCRF' library is required. Please install it using 'pip install torch-crf'.")
    exit()

# ========== CONFIG ==========
DATA_DIR = "output_data"
os.makedirs(DATA_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CHAR_LEN = 16
EMBED_DIM = 100
CHAR_EMBED_DIM = 30
CHAR_CNN_OUT = 30
BBOX_DIM = 100
HIDDEN_SIZE = 512  # Retaining the increased hidden size from our discussion
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
# Must match the normalization constant used when generating the BBox data!
BBOX_NORM_CONSTANT = 1000.0
CHUNK_SIZE = 256  # The size used for chunking

# ========== LABELS ==========
LABELS = ["O", "B-QUESTION", "I-QUESTION", "B-OPTION", "I-OPTION", "B-ANSWER", "I-ANSWER", "B-IMAGE", "I-IMAGE"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}



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

    # --- CRITICAL FIX: Implement dictionary-like lookup ---
    def __getitem__(self, token: str) -> int:
        """Allows lookup using word_vocab[token]. Returns UNK index if token is not found."""
        return self.stoi.get(token, self.stoi[self.unk_token])

    # --- Pickling methods for serialization (Kept from last fix) ---
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

# ========== Data Loading ==========
def load_unified_data(unified_json_path: str) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
    """
    Loads data from the Unified JSON file and packages the flat list
    back into a page/document structure for the DataLoader.
    """
    if not os.path.exists(unified_json_path):
        raise FileNotFoundError(f"Unified JSON data not found at: {unified_json_path}")

    with open(unified_json_path, 'r', encoding='utf-8') as f:
        flat_tokens = json.load(f)

    pages_tokens = []
    labels_per_token = []

    for i in range(0, len(flat_tokens), CHUNK_SIZE):
        chunk = flat_tokens[i:i + CHUNK_SIZE]
        if not chunk: continue

        # Prepare tokens list for the Dataloader's expected format
        tokens_list = [
            {
                "text": t["token"],
                # We use the raw BBox values here, they get normalized in the Dataset
                "x0": t["bbox"][0], "y0": t["bbox"][1],
                "x1": t["bbox"][2], "y1": t["bbox"][3],
                "page_no": 0, "block_idx": 0
            }
            for t in chunk
        ]

        pages_tokens.append({"tokens": tokens_list, "width": BBOX_NORM_CONSTANT, "height": BBOX_NORM_CONSTANT})
        labels_per_token.append([t["label"] for t in chunk])

    return pages_tokens, labels_per_token


# ========== Dataset ==========
class MCQTokenDataset(Dataset):
    def __init__(self, pages_tokens, word_vocab, char_vocab, labels_per_token=None):
        self.samples = []
        self.bbox_norm_factor = BBOX_NORM_CONSTANT

        for page_data in pages_tokens:
            if len(page_data["tokens"]) == 0: continue
            self.samples.append(page_data)

        self.labels = labels_per_token
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        page_data = self.samples[idx]
        toks = page_data["tokens"]

        words = [t["text"] for t in toks]
        word_ids = [self.word_vocab.stoi.get(w, self.word_vocab.stoi[self.word_vocab.unk_token]) for w in words]

        char_ids = []
        for w in words:
            chs = [self.char_vocab.stoi.get(ch, self.char_vocab.stoi[self.char_vocab.unk_token]) for ch in
                   w[:MAX_CHAR_LEN]]
            if len(chs) < MAX_CHAR_LEN:
                chs += [self.char_vocab.stoi[self.char_vocab.pad_token]] * (MAX_CHAR_LEN - len(chs))
            char_ids.append(chs)

        # Read and normalize REAL Bounding Box data
        bboxes = []
        for t in toks:
            normalized_bbox = [
                t["x0"] / self.bbox_norm_factor,
                t["y0"] / self.bbox_norm_factor,
                t["x1"] / self.bbox_norm_factor,
                t["y1"] / self.bbox_norm_factor,
            ]
            bboxes.append(normalized_bbox)

        labels = None
        if self.labels:
            lbls = self.labels[idx]
            labels = [LABEL2IDX[l] for l in lbls]

        return {
            "word_ids": torch.LongTensor(word_ids),
            "char_ids": torch.LongTensor(char_ids),
            "bboxes": torch.FloatTensor(bboxes),
            "labels": torch.LongTensor(labels) if labels is not None else None,
            "tokens": toks
        }


def collate_batch(batch):
    max_len = max(item["word_ids"].size(0) for item in batch)
    batch_size = len(batch)
    word_pad = torch.zeros((batch_size, max_len), dtype=torch.long)
    char_pad = torch.zeros((batch_size, max_len, MAX_CHAR_LEN), dtype=torch.long)
    bbox_pad = torch.zeros((batch_size, max_len, 4), dtype=torch.float)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    label_pad = torch.full((batch_size, max_len), -1, dtype=torch.long)
    tokens_list = []

    for i, item in enumerate(batch):
        L = item["word_ids"].size(0)
        word_pad[i, :L] = item["word_ids"]
        char_pad[i, :L, :] = item["char_ids"]
        bbox_pad[i, :L, :] = item["bboxes"]
        mask[i, :L] = 1
        if item["labels"] is not None:
            label_pad[i, :L] = item["labels"]
        tokens_list.append(item["tokens"])

    return {
        "words": word_pad,
        "chars": char_pad,
        "bboxes": bbox_pad,
        "mask": mask,
        "labels": label_pad,
        "tokens": tokens_list
    }


# ========== MODEL (With 2-layer LSTM and Dropout) ==========
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

        # 2 layers and 0.3 dropout to combat overfitting/plateauing
        self.bilstm = nn.LSTM(in_dim, HIDDEN_SIZE // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.ff = nn.Linear(HIDDEN_SIZE, n_labels)
        self.crf = CRF(n_labels)
        self.dropout = nn.Dropout(p=0.5)  # Input dropout

    def forward_emissions(self, words, chars, bboxes, mask):
        wemb = self.word_emb(words)
        cenc = self.char_enc(chars)
        benc = self.bbox_proj(bboxes)
        enc_in = torch.cat([wemb, cenc, benc], dim=-1)
        enc_in = self.dropout(enc_in)

        lengths = mask.sum(dim=1).cpu()

        packed_in = nn.utils.rnn.pack_padded_sequence(enc_in, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed_in)
        padded_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        return self.ff(padded_out)

    def forward(self, words, chars, bboxes, mask, labels=None, class_weights=None, alpha=0.7):
        emissions = self.forward_emissions(words, chars, bboxes, mask)

        if labels is not None:
            crf_loss = -self.crf(emissions, labels, mask=mask).sum()
            if class_weights is not None:
                ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(emissions.device), ignore_index=-1)
                ce_loss = ce_loss_fn(emissions.view(-1, emissions.size(-1)), labels.view(-1))
                return alpha * crf_loss + (1 - alpha) * ce_loss
            return crf_loss

        return self.crf.viterbi_decode(emissions, mask=mask)


# ========== training/eval (Eval uses Viterbi output correctly) ==========
def compute_class_weights(labels_list, num_labels):
    all_labels_flat = [lbl for page in labels_list for lbl in page]
    counts = Counter(all_labels_flat)
    total = sum(counts.values())
    weights = []
    for i in range(num_labels):
        count = counts.get(i, 0)
        # Apply inverse frequency weighting
        w = total / (num_labels * count) if count > 0 else 1.0
        # Boost specific categories (if needed, adjust these weights)
        if IDX2LABEL[i] in ["B-QUESTION", "B-OPTION"]:
            w *= 2.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float)


def eval_model(model, data_loader):
    # ... (evaluation code remains the same)
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in data_loader:
            words, chars, bboxes, mask = (batch[k].to(DEVICE) for k in ["words", "chars", "bboxes", "mask"])
            labels = batch["labels"].to(DEVICE)

            preds_batch = model(words, chars, bboxes, mask, labels=None)

            for i in range(len(preds_batch)):
                L = len(preds_batch[i])
                all_pred.extend(preds_batch[i])
                all_true.extend(labels[i][:L].cpu().numpy().tolist())

    from sklearn.metrics import precision_recall_fscore_support
    # Note: We use average='micro' for sequence tagging F1 score, which treats all token predictions equally.
    return precision_recall_fscore_support(all_true, all_pred, average='micro', zero_division=0)


def train_model(model, train_loader, val_loader, epochs=EPOCHS, class_weights=None):
    # ... (training loop remains the same)
    model.to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_f1 = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train E{ep}"):
            optim.zero_grad()
            words, chars, bboxes, mask, labels = (batch[k].to(DEVICE) for k in
                                                  ["words", "chars", "bboxes", "mask", "labels"])
            loss = model(words, chars, bboxes, mask, labels, class_weights=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {ep} train loss {avg_loss:.4f}")
        p, r, f1, _ = eval_model(model, val_loader)
        print(f"VAL p={p:.4f} r={r:.4f} f1={f1:.4f}")
        if f1 > best_val_f1:
            best_val_f1 = f1
            # Only save the model if it's the best performing one
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "model_CAT.pt"))
    print("Training complete. Best val F1:", best_val_f1)


# ========== helpers (FIXED: Added robust checks) ==========
def build_vocabs(train_pages_tokens):
    """Builds and verifies vocabs using only the training data."""
    word_vocab = Vocab(min_freq=1)
    char_vocab = Vocab(min_freq=1, unk_token="<CUNK>", pad_token="<CPAD>")

    for p in train_pages_tokens:
        for tok in p["tokens"]:
            text_value = tok["text"]
            word_vocab.add_sentence([text_value])
            char_vocab.add_sentence(list(text_value[:MAX_CHAR_LEN]))

    # CRUCIAL: Call build() on both vocabs to populate the indices
    word_vocab.build()
    char_vocab.build()

    # CRITICAL CHECK: Ensure the vocabulary was actually built
    if len(word_vocab) <= 2:
        raise ValueError(f"FATAL: Word vocabulary size is only {len(word_vocab)}. "
                         "Check your input JSON data to ensure tokens were extracted.")

    return word_vocab, char_vocab


def save_vocabs(path, word_vocab, char_vocab):
    with open(path, "wb") as f:
        pickle.dump((word_vocab, char_vocab), f)


def convert_labels_to_indices(all_labels):
    return [[LABEL2IDX[l] for l in page] for page in all_labels]


# ========== main training function (FIXED: Added final debug check) ==========
def train_from_json(unified_json_path: str):
    print("📥 Loading unified layout-aware labeled data...")
    all_pages_tokens, all_labels = load_unified_data(unified_json_path)

    if not all_labels:
        raise RuntimeError("❌ No labeled data found. Please check your unified JSON file.")

    print(f"📊 Total dataset size: {len(all_labels)} samples (chunks)")

    split_idx = int(len(all_pages_tokens) * 0.8)
    train_pages_tokens = all_pages_tokens[:split_idx]
    train_labels = all_labels[:split_idx]
    val_pages_tokens = all_pages_tokens[split_idx:]
    val_labels = all_labels[split_idx:]

    print(f"✅ Training on {len(train_labels)} samples, validating on {len(val_labels)} samples")

    all_labels_indices = convert_labels_to_indices(all_labels)
    class_weights = compute_class_weights(all_labels_indices, len(LABELS)).to(DEVICE)
    print("📏 Class weights:", class_weights)

    # 1. Build and verify the vocabs
    word_vocab, char_vocab = build_vocabs(train_pages_tokens)

    # 2. Add final debug print before saving (for your verification)
    print(f"DEBUG TRAINING: Final word vocab size before saving: {len(word_vocab)}")

    # 3. Save the verified vocabs
    save_vocabs(os.path.join(DATA_DIR, "vocabs_CAT.pkl"), word_vocab, char_vocab)

    dataset_train = MCQTokenDataset(train_pages_tokens, word_vocab, char_vocab, labels_per_token=train_labels)
    dataset_val = MCQTokenDataset(val_pages_tokens, word_vocab, char_vocab, labels_per_token=val_labels)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    model = MCQTagger(len(word_vocab), len(char_vocab), len(LABELS))
    print("🚀 Starting training...")
    train_model(model, train_loader, val_loader, epochs=EPOCHS, class_weights=class_weights)

    print("\n✅ Training complete.")
    print(f"📦 Model saved to: {os.path.join(DATA_DIR, 'model_CAT.pt')}")
    print(f"📦 Vocabularies saved to: {os.path.join(DATA_DIR, 'vocabs_CAT.pkl')}")


if __name__ == "__main__":
    # ⚠️ UPDATE THIS PATH to the file generated by the alignment script
    UNIFIED_DATA_PATH = "unified_training_data_bluuhhhhh.json"

    try:
        train_from_json(UNIFIED_DATA_PATH)
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Please ensure your data preprocessing step has created the 'unified_training_data.json' file.")
    except ValueError as e:
        print(f"\nFATAL ERROR: {e}")