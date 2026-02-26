

import os
import pickle
import fasttext
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# ============================================================
# TEMPERATURE SCALING HELPERS
# ============================================================

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def temperature_scale(probs, T=1.8):
    """Reduces overconfidence. T > 1 spreads probabilities."""
    logits = np.log(np.clip(probs, 1e-9, 1))
    scaled = logits / T
    return softmax(scaled)

# ============================================================
# LOAD PATHS
# ============================================================

SUBJECT_MODEL_FILE = "subject_classifier_lsvc.pkl"
TFIDF_VECTORIZER_FILE = "tfidf_vectorizer_ngram.pkl"
SCALER_FILE = "scaler.pkl"
CONCEPT_MAP_FILE = "concept_model_map.pkl"
FASTTEXT_MODEL_PATH_FILE = "fasttext_model_path.txt"

# ============================================================
# FASTTEXT + TF-IDF EMBEDDING
# ============================================================

class RealFastTextToVectorConverter:
    def __init__(self, model_path, tfidf_vectorizer):
        self.model = fasttext.load_model(model_path)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.vector_size = self.model.get_dimension()

        features = tfidf_vectorizer.get_feature_names_out()
        idf = tfidf_vectorizer.idf_
        self.idf_map = dict(zip(features, idf))
        self.tokenizer = tfidf_vectorizer.build_tokenizer()

    def transform_one(self, text):
        tokens = self.tokenizer(text.lower())

        uni_vec = np.zeros(self.vector_size, dtype=np.float32)
        uni_w = 0.0
        for w in tokens:
            w_idf = self.idf_map.get(w, 1.0)
            uni_vec += self.model.get_word_vector(w) * w_idf
            uni_w += w_idf
        if uni_w > 0:
            uni_vec /= uni_w

        bi_vec = np.zeros(self.vector_size, dtype=np.float32)
        bi_w = 0.0
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i+1]
            bigram = f"{w1} {w2}"
            bi_idf = self.idf_map.get(bigram, 1.0)
            pair = 0.5 * (self.model.get_word_vector(w1) + self.model.get_word_vector(w2))
            bi_vec += pair * bi_idf
            bi_w += bi_idf
        if bi_w > 0:
            bi_vec /= bi_w

        return np.concatenate([uni_vec, bi_vec], axis=0).reshape(1, -1)

# ============================================================
# LOAD MODELS
# ============================================================

print("Loading TF-IDF vectorizer...")
tfidf = pickle.load(open(TFIDF_VECTORIZER_FILE, "rb"))

print("Loading scaler...")
scaler = pickle.load(open(SCALER_FILE, "rb"))

print("Loading subject classifier...")
subject_model = pickle.load(open(SUBJECT_MODEL_FILE, "rb"))

print("Loading concept model map...")
concept_map = pickle.load(open(CONCEPT_MAP_FILE, "rb"))

fasttext_path = open(FASTTEXT_MODEL_PATH_FILE).read().strip()
print(f"Loading FastText: {fasttext_path}")
embedder = RealFastTextToVectorConverter(fasttext_path, tfidf)

print("Loading LightGBM concept models...")
concept_models = {}
for subject, meta in concept_map.items():
    with open(meta["file"], "rb") as f:
        concept_models[subject] = pickle.load(f)

# ============================================================
# CALIBRATED SUBJECT PROBS
# ============================================================

def get_subject_probs(model, X):
    """
    Returns calibrated subject probabilities.
    """
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)[0]
    else:
        # fallback
        pred = model.predict(X)[0]
        labels = model.classes_
        raw = np.array([1.0 if l == pred else 0.0 for l in labels])

    # --- Temperature scaling (reduces overconfidence) ---
    calibrated = temperature_scale(raw, T=1.8)

    labels = model.classes_
    ranked = sorted(zip(labels, calibrated), key=lambda x: x[1], reverse=True)
    return ranked

# ============================================================
# CALIBRATED CONCEPT PROBS
# ============================================================

def get_concept_probs(lgbm_container, X):
    """
    LightGBM → probabilities → smoothing → temperature scaling.
    """
    model = lgbm_container["model"]
    id_to_concept = lgbm_container["id_to_concept"]

    raw = model.predict_proba(X)[0]

    # smoothing + normalization
    raw = (raw + 1e-6)
    raw = raw / raw.sum()

    # temperature scaling
    calibrated = temperature_scale(raw, T=2.0)

    ranked = sorted(
        [(id_to_concept[i], calibrated[i]) for i in range(len(calibrated))],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked

# ============================================================
# INFERENCE FUNCTION
# ============================================================

def predict(text):
    ft_vec = embedder.transform_one(text)
    extra = np.array([[len(text.split())]], dtype=np.float32)
    feats = np.concatenate([ft_vec, extra], axis=1)
    feats_scaled = scaler.transform(feats)

    subject_ranked = get_subject_probs(subject_model, feats_scaled)
    top_subject = subject_ranked[0][0]

    if top_subject not in concept_models:
        return {
            "subject_top3": subject_ranked[:3],
            "concept_top5": [("UNKNOWN", 0.0)]
        }

    concept_ranked = get_concept_probs(concept_models[top_subject], feats_scaled)

    return {
        "subject_top3": subject_ranked[:3],
        "concept_top5": concept_ranked[:5]
    }

# ============================================================
# INTERACTIVE CLI
# ============================================================

if __name__ == "__main__":
    print("\n Hierarchical Inference Ready with Calibration!")
    print("Type a sentence to classify (or 'exit'):\n")

    while True:
        text = input("Enter text: ").strip()
        if text.lower() in ["exit", "quit"]:
            break

        out = predict(text)

        print("\n--- SUBJECT TOP 3 ---")
        for s, p in out["subject_top3"]:
            print(f"{s}: {p:.4f}")

        print("\n--- CONCEPT TOP 5 ---")
        for c, p in out["concept_top5"]:
            print(f"{c}: {p:.4f}")

        print("\n")

