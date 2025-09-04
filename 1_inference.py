import os
import glob
import shutil
import torch
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils import tokens_tags_to_spans
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Run inference with a selected model.")
parser.add_argument(
    "--dir_model_name",
    type=str,
    choices=[
        "camembert-bio",
        "camembert-base",
        "flaubert-base",
        "bert-base-multilingual",
        "fr-albert"
    ],
    help="Choose a model name from the available options."
)
parsargs = parser.parse_args()

dir_model_name = parsargs.dir_model_name
print(f"Using model: {dir_model_name}")

# Directories
PROJECT_ROOT = Path.cwd()
MODEL_PATH = PROJECT_ROOT / "models" / dir_model_name
INPUT_DIR = PROJECT_ROOT / "data/brat_processed"
OUTPUT_DIR = PROJECT_ROOT / "inference" / dir_model_name

for p in [MODEL_PATH, INPUT_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Load spaCy
nlp = spacy.load("fr_core_news_sm")

def predict_tags(model, tokenizer, words, device):
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds    = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
    word_ids = inputs.word_ids(batch_index=0)

    tags = []
    prev = None
    for p, wid in zip(preds, word_ids):
        if wid is None or wid == prev:
            prev = wid
            continue
        tags.append(model.config.id2label[p])
        prev = wid
    return tags

def main():
    txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    for txt_path in tqdm(txt_files, desc="Processing files"):

        text = open(txt_path, encoding="utf-8").read()
        doc  = nlp(text)

        spans = []
        for sent in doc.sents:
            tokens = [w for w in sent if not w.is_space]
            words  = [w.text for w in tokens]
            tags   = predict_tags(model, tokenizer, words, device)
            spans.extend(tokens_tags_to_spans(text, tokens, tags))

        # copy .txt
        basename = os.path.basename(txt_path)
        shutil.copy(txt_path, os.path.join(OUTPUT_DIR, basename))

        # write .ann
        ann_path = os.path.join(OUTPUT_DIR, basename.replace(".txt", ".ann"))
        with open(ann_path, "w", encoding="utf-8") as f:
            for idx, (label, start, end, span_text) in enumerate(spans, start=1):
                f.write(f"T{idx}\t{label} {start} {end}\t{span_text}\n")

if __name__ == "__main__":
    main()
