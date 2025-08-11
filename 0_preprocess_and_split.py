# preprocess_and_split.py

from src.brat_utils import load_ner_dataframe, split_and_save

INPUT_DIR  = "data/brat_format"
OUTPUT_DIR = "data/csv_format"

df = load_ner_dataframe(INPUT_DIR)
train_df, test_df = split_and_save(df, OUTPUT_DIR)
print(f"Saved {len(train_df)} train and {len(test_df)} test sentences.")