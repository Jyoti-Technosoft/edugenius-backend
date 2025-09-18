
import torch
from TorchCRF import CRF
import json
import numpy as np
import evaluate
from datasets import Dataset, Features, Value, Sequence
from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
from typing import List, Dict, Any


# --- Custom Model with CRF Layer ---
class DistilBertCrfForTokenClassification(nn.Module):
    def __init__(self, num_labels, model_name="distilbert-base-uncased"):
        super(DistilBertCrfForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.distilbert.config.seq_classif_dropout)
        self.classifier = nn.Linear(self.distilbert.config.dim, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            # Create a mask to ignore the -100 labels
            crf_mask = labels != -100
            # Replace -100 with a valid label, e.g., 0, so CRF doesn't error out.
            # The mask ensures these positions are not considered in the loss.
            labels[labels == -100] = 0

            # Sum the loss over the batch to get a scalar value for backpropagation
            crf_loss = -self.crf(logits, labels, crf_mask).sum()
            return {"loss": crf_loss, "logits": logits}

        return {"logits": logits}

    def predict(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        # Use Viterbi decoding to get the best tag sequence
        viterbi_path = self.crf.viterbi_decode(logits, attention_mask.bool())

        return viterbi_path


# --- Data Loading and Pre-processing (from your original script) ---
with open('poopy_butthole.json', 'r', encoding='utf-8') as f:
    uploaded_data = json.load(f)

sentences = []
labels = []
current_sentence_tokens = []
current_sentence_labels = []
for item in uploaded_data:
    if item['label'].startswith('B-QUESTION') and current_sentence_tokens:
        sentences.append(current_sentence_tokens)
        labels.append(current_sentence_labels)
        current_sentence_tokens = []
        current_sentence_labels = []

    current_sentence_tokens.append(item['token'])
    current_sentence_labels.append(item['label'])

    if item['label'].startswith('B-ANSWER'):
        sentences.append(current_sentence_tokens)
        labels.append(current_sentence_labels)
        current_sentence_tokens = []
        current_sentence_labels = []

if current_sentence_tokens:
    sentences.append(current_sentence_tokens)
    labels.append(current_sentence_labels)

unique_labels = sorted(list(set(l for sentence_labels in labels for l in sentence_labels)))
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for i, label in enumerate(unique_labels)}

features = Features({
    'tokens': Sequence(Value('string')),
    'ner_tags': Sequence(Value('int64'))
})
hf_dataset_dict = {
    'tokens': sentences,
    'ner_tags': [[label_to_id[label] for label in sentence_labels] for sentence_labels in labels]
}
raw_dataset = Dataset.from_dict(hf_dataset_dict, features=features)

train_test_split = raw_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)


# --- Custom Data Collator for Padding ---
class DataCollatorForTokenClassification:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]):
        # Separate the labels from the other features
        labels = [f['labels'] for f in features]
        # Remove labels and tokens from the features
        inputs = [{k: v for k, v in f.items() if k in ['input_ids', 'attention_mask']} for f in features]

        # Pad the inputs using the tokenizer
        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )

        # Manually pad the labels to the same length as input_ids
        max_length = batch["input_ids"].size(1)
        padded_labels = []
        for label_list in labels:
            padding_length = max_length - len(label_list)
            padded_labels.append(label_list + [-100] * padding_length)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


data_collator = DataCollatorForTokenClassification(tokenizer)

# --- Custom Training Loop ---
model = DistilBertCrfForTokenClassification(num_labels=len(unique_labels))
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=16, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=16, collate_fn=data_collator)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Training)")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} finished. Avg Training Loss: {avg_train_loss:.4f}")

    # Evaluation loop
    model.eval()
    metric = evaluate.load("seqeval")
    progress_bar_eval = tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Evaluation)")

    for batch in progress_bar_eval:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            predictions = model.predict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        true_labels_for_eval = []
        true_predictions_for_eval = []
        for labels, preds in zip(batch["labels"].tolist(), predictions):
            valid_labels = []
            valid_preds = []
            for label, pred in zip(labels, preds):
                if label != -100:
                    valid_labels.append(id_to_label[label])
                    valid_preds.append(id_to_label[pred])
            true_labels_for_eval.append(valid_labels)
            true_predictions_for_eval.append(valid_preds)

        metric.add_batch(predictions=true_predictions_for_eval, references=true_labels_for_eval)

    eval_results = metric.compute()
    print("Evaluation Results:")
    print(eval_results)

# Save the trained model and tokenizer
output_dir = "./CRF_BERT_MODEL"
torch.save(model.state_dict(), f"{output_dir}/model.pt")
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")