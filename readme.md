# French Oncology NER Corpus – Training and Inference

This repository provides code and resources to train and evaluate named entity recognition (NER) models on a French oncology corpus annotated with ICD-O-3 categories (morphologie, topographie, différenciation, expression_CIM).

## Repository structure
```
data/
├─ brat_raw/         Manually annotated corpus in BRAT standoff format (.txt + .ann)
├─ brat_processed/   Preprocessed version for fine-tuning 
├─ csv/              BIO-formatted annotations in CSV
└─ tsv/              BIO-formatted annotations in TSV

0_train.py           Fine-tune BERT-based models on the dataset
1_inference.py       Run inference and export predictions in BRAT format
utils.py             Data processing and helper functions
requirements.txt     Python dependencies
```
## Dataset statistics

### `brat_raw/` 
- Manually annotated corpus in BRAT standoff format. Contains both entity annotations and their corresponding ICD-O-3 code.  
- **Total entities**: 71,346  
- **Total texts**: 1,301 synthetic French clinical cases. Provided as `.txt` files, each paired with an `.ann` file containing the annotations. 

### Entity distribution in `brat_raw/`

| Label         | Count (n, %)          | Unique ICD-O codes       |
|------------------|-----------------------|--------------------------|
| Expression_CIM   | 25,738 (36.1%)        | \>1,500 combinations     |
| Morphologie      | 25,697 (36.0%)        | ~350 codes               |
| Topographie      | 18,864 (26.4%)        | ~300 codes               |
| Différenciation  | 1,047 (1.5%)          | 4 codes                  |




### `brat_processed/`
- Contains a preprocessed version of the corpus prepared for fine-tuning workflows.  
- Overlapping `expression_CIM` annotations are removed, leaving three categories (morphologie, topographie, différenciation). Does not contain ICD-O-3 codes, only entity spans.
- **Total entities**: 46,468
- **Total texts**: 1,301 synthetic French clinical cases, following the same `.txt`/`.ann` file structure and filename prefixes as `brat_raw/`.  
 

### Entity distribution in `brat_processed/`

| Label         | Count (n, %)      |
|------------------|-------------------|
| Morphologie      | 26,480 (57.0%)    |
| Topographie      | 18,939 (40.8%)    |
| Différenciation  | 1,049 (2.2%)      |






### Entity counts

| Dataset         | Total entities | Morphologie | Topographie | Différenciation | Expression_CIM |
|-----------------|----------------|-------------|-------------|-----------------|----------------|
| **brat_raw/**   | 71,346         | 25,697 (36.0%)      | 18,864 (26.4%)       | 1,047 (1.5%)            | 25,738 (36.1%)          |
| **brat_processed/** | 46,468    | 26,480 (57.0%)      | 18,939 (40.8%)       | 1,049 (2.26%)         | –              |

- **brat_raw/**: full annotation layer including overlapping `expression_CIM` spans  
- **brat_processed/**: NER-compatible version with `expression_CIM` removed and no discontinuous entities


## Installation

Create a Python environment (Python ≥3.9 recommended) and install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

## Training and inference 
To fine-tune a model, run : 
```bash
python 0_train.py --dir_model_name <model name>
```

To apply a trained model to the processed corpus and output BRAT-style annotations, run :

```bash
python 1_inference.py --dir_model_name <model name>
```
Predictions will be written in inference/<model_name>/ as .txt and .ann files. 

Available options for `--dir_model_name`:

- `camembert-bio`  
- `camembert-base`  
- `bert-base-multilingual`  
- `fr-albert`  
- `xlm-roberta`

All hyperparameters (batch size, learning rate, epochs, etc.) were kept constant across experiments.  
To reproduce the published baselines, it is sufficient to vary only `--dir_model_name`.  


