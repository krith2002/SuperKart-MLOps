import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "SuperKart.csv"
PROCESSED_DIR = DATA_DIR / "processed"


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw data shape: {df.shape}")
    return df


def clean_data(df):
    df = df.copy()

    # Remove identifier-like columns not useful for modeling
    cols_to_drop = ["Product_Id", "Store_Id"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Target for this dataset is the numeric sales total
    target_col = "Product_Store_Sales_Total"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col]).copy()

    # Fill numeric missing values with median and categorical with mode
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode_value = df[column].mode(dropna=True)
            if not mode_value.empty:
                df[column] = df[column].fillna(mode_value.iloc[0])
            else:
                df[column] = df[column].fillna("Unknown")

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def save_split_data(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Saved train shape: {train_df.shape}")
    print(f"Saved test shape: {test_df.shape}")


def main():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}. Please place SuperKart.csv inside data/raw/")

    df = load_data()
    cleaned_df = clean_data(df)
    save_split_data(cleaned_df)

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
