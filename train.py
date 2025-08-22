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
from argparse import Namespace
from IPython.display import display
import evaluate
import json
import numpy as np 
from datetime import datetime


PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
TRAIN_CSV_FILE = PROJECT_ROOT / "data/csv_format/ner_sentences_train.csv"
TEST_CSV_FILE  = PROJECT_ROOT / "data/csv_format/ner_sentences_test.csv"

train_dataset = dataset_generator(TRAIN_CSV_FILE)
test_dataset = dataset_generator(TEST_CSV_FILE)

# Get label list and mapping to ID
label_list = train_dataset["train"].features[f"tags"].feature.names
# print("Labels: ", label_list)
label2id = {label: id for id, label in enumerate(label_list)}
id2label = {id: label for label, id in label2id.items()}

# Check if running on GPU
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


# model options
model_options = [
    "almanach/camembert-bio-base",
    "camembert-base",
    "flaubert/flaubert_base_cased",
    # "Dr-BERT/DrBERT-7GB",
    "bert-base-multilingual-cased", 
    "cservan/french-albert-base-cased"
]

# model default hyperparameters
model_args_dict = {
    "model_name":"almanach/camembert-bio-base",
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps":2, 
    "learning_rate": 5e-5,
    "remove_unused_columns": True,
    "seed": 42,
    "logging_dir": "logs",
    "logging_steps": 100,           # log every N steps
    "report_to": ["tensorboard"],  # log to TensorBoard
    "load_best_model_at_end": True,
    "save_total_limit": 2,  # keep only the last 2 checkpoints
    "metric_for_best_model": "f1",
    "greater_is_better": True,  # for f1, higher is better
    "eval_strategy": "epoch",            # evaluate each epoch
    "save_strategy": "epoch",          # checkpoint each epoch
}

model_args = Namespace(**model_args_dict)


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(model_args.model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id)
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

# Define output and model location
MODEL_NAME_MAP = {
    "almanach/camembert-bio-base": "camembert-bio",
    "camembert-base": "camembert-base",
    "flaubert/flaubert_base_cased": "flaubert-base",
    "bert-base-multilingual-cased": "bert-base-multilingual",
    "cservan/french-albert-base-cased": "fr-albert"
}
dir_model_name = MODEL_NAME_MAP[model_args.model_name]

# Create output directories
OUTPUT_DIR_HPO = PROJECT_ROOT / "runs/hpo" / f"{dir_model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
OUTPUT_DIR_FINAL = PROJECT_ROOT / "runs/final_best" / f"{dir_model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / dir_model_name

OUTPUT_DIR_HPO.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_FINAL.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Hyperparameter optimization setup
model_init = partial(
    build_token_cls,
    model_args.model_name,
    len(label_list),
    id2label,
    label2id
)

hpo_args = TrainingArguments(
    output_dir=OUTPUT_DIR_HPO, 
    eval_strategy="epoch",
    save_strategy="best",
    load_best_model_at_end=True,
    logging_strategy="epoch",
    report_to="none",
    metric_for_best_model=model_args.metric_for_best_model,
    greater_is_better=model_args.greater_is_better,
    seed=42, data_seed=42,   # fix seed during HPO; vary seeds later
)

hpo_trainer = Trainer(
    args=hpo_args,
    model_init=model_init,
    train_dataset=split_train_dataset["train"],
    eval_dataset=split_train_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=partial(compute_metrics, label_list=label_list)  # must return "f1", "precision", "recall", etc.
)

best_run = hpo_trainer.hyperparameter_search(
    direction="maximize",
    hp_space=hp_space,
    compute_objective=partial(compute_objective, label_list=label_list),
    n_trials=20,
    backend="optuna",
    # persist:
    study_name=f"{dir_model_name}_hpo",
    storage=f"sqlite:///{OUTPUT_DIR_HPO}/optuna.db",
    load_if_exists=True,
)
print("Best hyperparameters:", best_run.hyperparameters)

with open(OUTPUT_DIR_HPO / "best_hparams.json", "w") as f:
    json.dump(best_run.hyperparameters, f, indent=2)



# Run five random initializations and log results to TB
seeds = [11, 22, 33, 44, 55] # should i make this random?

val_results, test_results = [], []
for s in seeds:
    best_args = TrainingArguments(
        output_dir=os.fspath(OUTPUT_DIR_FINAL / f"s{s}"),
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model=model_args.metric_for_best_model,
        greater_is_better=True,
        report_to=model_args.report_to,
        seed=s,                   # <- governs init/dropout/etc.
        data_seed=s,              # <- governs shuffling/samplers
        remove_unused_columns=model_args.remove_unused_columns,
        per_device_train_batch_size=best_run.hyperparameters.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=max(16, best_run.hyperparameters.get("per_device_train_batch_size", 8)*4),
        gradient_accumulation_steps=best_run.hyperparameters.get("gradient_accumulation_steps", 2),
        learning_rate=best_run.hyperparameters.get("learning_rate", 5e-5),
        num_train_epochs=best_run.hyperparameters.get("num_train_epochs", 5),
        weight_decay=best_run.hyperparameters.get("weight_decay", 0.01),
        warmup_ratio=best_run.hyperparameters.get("warmup_ratio", 0.1),
        save_total_limit=model_args.save_total_limit,
        logging_steps=model_args.logging_steps,
        logging_dir=model_args.logging_dir,
        load_best_model_at_end=model_args.load_best_model_at_end,
        run_name=f"{dir_model_name}_s{s}",  # TB run label
    )

    final_trainer = Trainer(
        args=best_args,
        model_init=model_init,   # ensures model is (re)initialized *after* seeding
        train_dataset=split_train_dataset["train"],
        eval_dataset=split_train_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, label_list=label_list),   # returns "f1", etc.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-4)]
    )

    final_trainer.train()

    # metrics on validation split per seed 
    val_metrics = final_trainer.evaluate() # metrics on validation split per seed 
    val_metrics["seed"] = s
    val_results.append(val_metrics)

    # metrics on test set per seed 
    test_metrics = final_trainer.evaluate(eval_dataset=tokenized_test_dataset["train"])
    test_metrics["seed"] = s
    test_results.append(test_metrics)

    # Save the final model
    seed_dir = MODEL_SAVE_PATH / f"seed_{s}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    final_trainer.save_model(seed_dir)  # saves model + config + tokenizer

    # Save run metadata alongside weights:
    with open(seed_dir / "metrics_val.json", "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(seed_dir / "metrics_test.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(seed_dir / "training_args.json", "w") as f:
        f.write(final_trainer.args.to_json_string())



