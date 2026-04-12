
import ast
import os
import pandas as pd

from normalized_csv.hand_normalization import normalize_hands_full
from normalized_csv.body_normalization import normalize_body_full


def preprocess(input_csv: str, output_csv: str):
    """
    Normalizes raw landmark CSV data (hands + body) and saves the result.

    :param input_csv: Path to the raw input CSV file.
    :param output_csv: Path where the normalized CSV will be saved.
    """
    df = pd.read_csv(input_csv, encoding="utf-8")

    # Temporary removed labels column
    labels = df["labels"].to_list()
    del df["labels"]

    convert = lambda x: ast.literal_eval(str(x))
    for column in df.columns:
        df[column] = df[column].apply(convert)

    df = normalize_hands_full(df)
    df, _ = normalize_body_full(df)

    df["labels"] = labels

    df.to_csv(output_csv, encoding="utf-8", index=False)
    print(f"Saved normalized data to: {output_csv}")


if __name__ == "__main__":
    DATASETS = [
        ("train_raw.csv", "train_normalized.csv"),
        ("test_raw.csv",  "test_normalized.csv"),
    ]
    for input_csv, output_csv in DATASETS:
        preprocess(input_csv, output_csv)
