from __future__ import annotations
from datasets import load_dataset, ClassLabel, Features, Sequence, Value
import ast 
import os
import numpy as np
import torch
import glob
from transformers import AutoModelForTokenClassification
import pathlib
import spacy
import evaluate
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
from itertools import zip_longest


nlp = spacy.load("fr_core_news_sm")
seqeval = evaluate.load("seqeval")

def parse_ann(path):
    anns = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("T"):
                continue
            _, meta, _ = line.strip().split("\t")
            label, spans = meta.split(" ", 1)
            spans = [tuple(map(int, s.split())) for s in spans.split(";")]
            anns.append({"label": label, "spans": spans})
    return anns

def get_word_tag(start, end, annotations):
    """
    Return:
      - "B-<label>" if this token begins exactly at a span boundary
      - "I-<label>" if it falls inside a span (but not at its start)
      - "O" otherwise
    """
    for ann in annotations:
        label = ann["label"]
        for s, e in ann["spans"]:
            if start >= s and end <= e:
                return ("B-" if start == s else "I-") + label
    return "O"




def load_ner_dataframe(input_dir):
    """
    Walk through all .txt/.ann pairs in input_dir,
    split into sentences, and return a DataFrame with columns ["words","tags"].
    If the .ann file is missing, it assumes there are no annotations.
    """
    rows = []
    txt_paths = glob.glob(os.path.join(input_dir, "*.txt"))
    for txt_path in tqdm(txt_paths, desc="Processing files"):
        text = open(txt_path, encoding="utf-8").read()

        ann_path = txt_path[:-4] + ".ann"
        if os.path.exists(ann_path):
            ann = parse_ann(ann_path)
        else:
            ann = []  # or {} depending on your parse_ann return type

        for sent in nlp(text).sents:
            words = [w.text for w in sent if not w.is_space]
            tags  = [get_word_tag(w.idx, w.idx + len(w), ann)
                     for w in sent if not w.is_space]
            if len(words) != len(tags):
                continue
            rows.append({"words": words, "tags": tags})
    return pd.DataFrame(rows)



def split_and_save(df, output_dir, test_size=0.2, random_state=42):
    """
    Split DataFrame into train/test, save to CSVs in output_dir.
    Returns (train_df, test_df).
    """
    train_df, test_df = train_test_split(df, test_size=test_size, 
                                         random_state=random_state)
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "ner_sentences_train.csv"),
                    index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(output_dir, "ner_sentences_test.csv"),
                   index=False, encoding="utf-8")
    return train_df, test_df


def tokens_tags_to_spans(text, tokens, tags):
    """
    Convert tokens and BIO-style tags into BRAT-style entity spans.
    Works even if tokens and tags lengths differ.
    """
    spans = []
    current_label = None
    current_start = None

    for token, tag in zip_longest(tokens, tags, fillvalue="O"):
        if tag is None:
            tag = "O"

        if tag.startswith("B-"):
            if current_label is not None:
                spans.append((
                    current_label,
                    current_start,
                    prev_end,
                    text[current_start:prev_end]
                ))
            current_label = tag[2:]
            current_start = token.idx
            prev_end = token.idx + len(token.text)

        elif tag.startswith("I-") and current_label == tag[2:]:
            prev_end = token.idx + len(token.text)

        else:
            if current_label is not None:
                spans.append((
                    current_label,
                    current_start,
                    prev_end,
                    text[current_start:prev_end]
                ))
                current_label = None
                current_start = None
            prev_end = token.idx + len(token.text)

    if current_label is not None:
        spans.append((
            current_label,
            current_start,
            prev_end,
            text[current_start:prev_end]
        ))

    return spans

def save_brat_predictions(txt_path, tokens, tags, output_dir):
    """
    Copy the original .txt to output_dir and write a .ann file
    with the spans computed from tokens+tags.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) copy the text
    basename = os.path.basename(txt_path)
    out_txt  = os.path.join(output_dir, basename)
    shutil.copy(txt_path, out_txt)

    # 2) read full text
    text = open(txt_path, encoding="utf-8").read()

    # 3) compute spans and write .ann
    spans   = tokens_tags_to_spans(text, tokens, tags)
    ann_path = os.path.join(output_dir, basename.replace(".txt", ".ann"))
    with open(ann_path, "w", encoding="utf-8") as fout:
        for idx, (label, start, end, span_text) in enumerate(spans, start=1):
            fout.write(f"T{idx}\t{label} {start} {end}\t{span_text}\n")
            

def convert_string_to_list(row):
    row["words"] = ast.literal_eval(row["words"])  
    row["tags"] = ast.literal_eval(row["tags"])    
    return row


def convert_tag_to_id(row, label_list):
    row["tags"] = [label_list.index(tag) for tag in row["tags"]]
    return row


def dataset_generator(data_files):
    if isinstance(data_files, pathlib.PurePath) : 
        data_files = str(data_files)
    dataset = load_dataset("csv", data_files=data_files)
    label_list = ["O", "B-morphologie", "I-morphologie", "B-topographie", "I-topographie", "B-differenciation", "I-differenciation"]
    dataset = dataset.map(convert_string_to_list)
    dataset = dataset.map(lambda row: convert_tag_to_id(row, label_list))
    features = Features({
        "words": Sequence(Value("string")),
        "tags": Sequence(ClassLabel(num_classes=len(label_list), names=label_list))
        })
    dataset = dataset.cast(features)
    return dataset


def get_tokenize_and_align_labels_fn(tokenizer, label2id=None, label_all_tokens=False):
    """
    Returns a tokenization function for token classification that handles both
    string and integer labels.

    Args:
        tokenizer: Hugging Face tokenizer
        label2id: optional dict mapping string tags to integer IDs
        label_all_tokens: if True, label all subtokens; else only the first subtoken
    Returns:
        Function that takes a batch of examples and returns tokenized inputs with aligned labels
    """
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["words"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, word_labels in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_val = word_labels[word_idx]

                    # Convert string labels to IDs if mapping is provided
                    if isinstance(label_val, str) and label2id:
                        label_val = label2id[label_val]

                    # Decide whether to label this subtoken
                    if word_idx != previous_word_idx:
                        label_ids.append(label_val)
                    else:
                        label_ids.append(label_val if label_all_tokens else -100)

                    previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return tokenize_and_align_labels


def build_token_cls(model_name: str, num_labels: int, id2label: dict, label2id: dict):
    return AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )


def compute_metrics(eval_pred, label_list, ignore_index=-100, _np=np, _seqeval=seqeval):
    # tolerant to EvalPrediction vs tuple
    logits = getattr(eval_pred, "predictions", eval_pred[0])
    labels = getattr(eval_pred, "label_ids",     eval_pred[1])

    if isinstance(logits, torch.Tensor): logits = logits.detach().float().cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()

    id2label = dict(enumerate(label_list))
    preds = _np.argmax(logits, axis=-1)
    mask  = labels != ignore_index

    preds_str  = [[id2label[int(p)] for p, m in zip(ps, ms) if m] for ps, ms in zip(preds,  mask)]
    labels_str = [[id2label[int(l)] for l, m in zip(ls, ms) if m] for ls, ms in zip(labels, mask)]

    # Guard 1: drop empty sequences
    pairs = [(p, r) for p, r in zip(preds_str, labels_str) if len(r) > 0]
    if not pairs:
        return {
            "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0
        }

    preds_str, labels_str = zip(*pairs)
    preds_str, labels_str = list(preds_str), list(labels_str)

    rep = _seqeval.compute(predictions=preds_str, references=labels_str, zero_division=0)

    metrics = {
        "f1":        rep.get("overall_f1", 0.0),
        "precision": rep.get("overall_precision", 0.0),
        "recall":    rep.get("overall_recall", 0.0),
        "accuracy":  rep.get("overall_accuracy", 0.0),
    }

    entities = sorted({lab.split("-", 1)[-1] for lab in label_list if lab != "O"})
    for ent in entities:
        stats = rep.get(ent, {"f1": 0.0, "precision": 0.0, "recall": 0.0, "number": 0})
        metrics[f"f1/{ent}"]        = stats.get("f1", 0.0)
        metrics[f"precision/{ent}"] = stats.get("precision", 0.0)
        metrics[f"recall/{ent}"]    = stats.get("recall", 0.0)
        metrics[f"support/{ent}"]   = stats.get("number", 0)

    return metrics