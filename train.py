from pathlib import Path
import random
from transformers.utils import logging
import torch
import os
from functools import partial
from transformers import (AutoTokenizer, DataCollatorForTokenClassification, 
                          AutoModelForTokenClassification, TrainingArguments, 
                          Trainer, EarlyStoppingCallback, set_seed) 
from utils import (dataset_generator, get_tokenize_and_align_labels_fn, 
                       compute_metrics, build_token_cls, hp_space, compute_objective)
import evaluate
import json
import argparse
import numpy as np 
from datetime import datetime

parser = argparse.ArgumentParser(description="Train a selected model.")
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


# If not provided, prompt interactively
if parsargs.dir_model_name is None:
    print("Please choose a model:")
    options = [
        "camembert-bio",
        "camembert-base",
        "flaubert-base",
        "bert-base-multilingual",
        "fr-albert"
    ]
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    choice = input("Enter the number corresponding to your choice [default=camembert-bio]: ")
    try:
        if choice.strip() == "":
            dir_model_name = "camembert-bio"  # default
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                dir_model_name = options[idx]
            else:
                raise ValueError
    except ValueError:
        raise SystemExit("Invalid choice. Exiting.")
else:
    dir_model_name = parsargs.dir_model_name

print(f"Using model: {dir_model_name}")

MODEL_NAME_MAP = {
    "camembert-bio": "almanach/camembert-bio-base",
    "camembert-base": "camembert-base",
    "flaubert-base": "flaubert/flaubert_base_cased",
    "bert-base-multilingual": "bert-base-multilingual-cased",
    "fr-albert": "cservan/french-albert-base-cased"
}

model_name = MODEL_NAME_MAP[dir_model_name]

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
TRAIN_CSV_FILE = PROJECT_ROOT / "data/csv_data/ner_sentences_train.csv"
TEST_CSV_FILE  = PROJECT_ROOT / "data/csv_data/ner_sentences_test.csv"

# Create output directories
OUTPUT_DIR = PROJECT_ROOT / "runs/final_best" / f"{dir_model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / f"{dir_model_name}_2"

for p in [OUTPUT_DIR, MODEL_SAVE_PATH]:
    p.mkdir(parents=True, exist_ok=True)

train_dataset = dataset_generator(TRAIN_CSV_FILE)
test_dataset = dataset_generator(TEST_CSV_FILE)

# Get label list and mapping to ID
label_list = train_dataset["train"].features[f"tags"].feature.names
label2id = {label: id for id, label in enumerate(label_list)}
id2label = {id: label for label, id in label2id.items()}

# Check if running on GPU (remove?)
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using:", device)


cfg = {
    "model_name": model_name,
    "seed_list": [11, 22, 33, 44, 55],
    "training_args": {
        # "num_train_epochs": 5,
        "num_train_epochs": 7,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        # "gradient_accumulation_steps": 2,
        "gradient_accumulation_steps": 1,
        # "learning_rate": 5e-5,
        "learning_rate": 3.894793689386536e-5,
        # "weight_decay": 0.01,
        "weight_decay" : 0.08061743396237347,
        # "warmup_ratio": 0.1,
        "warmup_ratio": 0.13860985884866567,
        "logging_dir": "logs",
        "logging_steps": 100,           # log every N steps
        "report_to": ["tensorboard"],   # log to TensorBoard
        "load_best_model_at_end": True,
        "save_total_limit": 2,          # keep only the last 2 checkpoints
        "metric_for_best_model": "f1",
        "greater_is_better": True,      # for f1, higher is better
        "eval_strategy": "epoch",       # evaluate each epoch
        "save_strategy": "epoch",       # checkpoint each epoch 
        # save_strategy : "best",  # save only the best model
        "remove_unused_columns": True,  # remove columns not used by the model
    }
}



# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(cfg["model_name"], num_labels=len(label_list), id2label=id2label, label2id=label2id)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer) 


# Tokenize datasets
tokenize_fn = get_tokenize_and_align_labels_fn(
    tokenizer=tokenizer,
    label2id=None,          # skip mapping because labels are already integers
    label_all_tokens=False
)

cols_to_remove = train_dataset["train"].column_names  # e.g., ['words','tags',...]
tokenized_train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove) 
tokenized_test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=cols_to_remove) 
split_train_dataset = tokenized_train_dataset["train"].train_test_split(test_size=0.1) # Split train dataset into train and validation sets 


model_init = partial(
    build_token_cls,
    cfg["model_name"],
    len(label_list),
    id2label,
    label2id
)

seeds = [11, 22, 33, 44, 55] # should i make this random?
val_results, test_results = [], []

for s in seeds:
    args = TrainingArguments(**cfg["training_args"], 
                             seed=s, data_seed=s, 
                             output_dir=os.fspath(OUTPUT_DIR / f"s{s}"), 
                             run_name=f"{dir_model_name}_s{s}")
    trainer = Trainer(
        args=args,
        model_init=model_init,   # ensures model is (re)initialized *after* seeding
        train_dataset=split_train_dataset["train"],
        eval_dataset=split_train_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, label_list=label_list),   # returns "f1", etc.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-4)]
    )

    trainer.train()

    # metrics on validation split per seed 
    val_metrics = trainer.evaluate() # metrics on validation split per seed 
    val_metrics["seed"] = s
    val_results.append(val_metrics)
    print(f"seed {s}: trained_epochs={trainer.state.epoch}")

    # metrics on test set per seed 
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test_dataset["train"])
    test_metrics["seed"] = s
    test_results.append(test_metrics)

    # Save the final model
    seed_dir = MODEL_SAVE_PATH / f"seed_{s}"
    if seed_dir.exists():
        raise SystemExit(f"{seed_dir} already exists; aborting to avoid overwrite.")
    seed_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(seed_dir)  # saves model + config + tokenizer

    # Save run metadata alongside weights:
    with open(seed_dir / "metrics_val.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(seed_dir / "metrics_test.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(seed_dir / "training_args.json", "w") as f:
        f.write(trainer.args.to_json_string())



