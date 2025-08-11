from datasets import load_dataset, ClassLabel, Features, Sequence, Value
import ast 
import os
import glob
import spacy
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
from itertools import zip_longest

nlp = spacy.load("fr_core_news_sm")

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
    """
    rows = []
    txt_paths = glob.glob(os.path.join(input_dir, "*.txt"))
    for txt_path in tqdm(txt_paths, desc="Processing files"):
        text = open(txt_path, encoding="utf-8").read()
        ann = parse_ann(txt_path[:-4] + ".ann")
        for sent in nlp(text).sents:
            words = [w.text for w in sent if not w.is_space]
            tags  = [get_word_tag(w.idx, w.idx + len(w), ann)
                     for w in sent if not w.is_space]
            if len(words) != len(tags):
                # skip malformed
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
    dataset = load_dataset("csv", data_files=data_files)
    label_list = ["O", "B-morphologie", "I-morphologie", "B-topographie", "I-topographie", "B-differenciation", "I-differenciation", "B-stade", "I-stade"]
    dataset = dataset.map(convert_string_to_list)
    dataset = dataset.map(lambda row: convert_tag_to_id(row, label_list))
    features = Features({
        "words": Sequence(Value("string")),
        "tags": Sequence(ClassLabel(num_classes=9, names=label_list))
        })
    dataset = dataset.cast(features)
    return dataset 


# if __name__ == "__main__":
#    dataset_generator()

# list = dataset["train"].features[f"tags"].feature.names
# print(list)
# print(dataset["train"].features)

# label_names = features['tags'].feature.names
# print(label_names)

# print(dataset['train'].features)

# list = dataset["train"].features[f"tags"].feature.names
# print(list)

# for i in range(10):
   # print(dataset["train"][i]) 