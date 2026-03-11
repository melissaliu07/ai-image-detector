import os
import shutil
import pandas as pd
from pathlib import Path

# Paths
RAW_DIR = "data/raw"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# Read CSVs
print("Reading CSV files...")
train_df = pd.read_csv(f"{RAW_DIR}/train.csv", 
                       header=None, 
                       names=["id", "filepath", "label"])
test_df = pd.read_csv(f"{RAW_DIR}/test.csv", 
                      header=None, 
                      names=["id", "filepath", "label"])

print(f"Train CSV: {len(train_df)} images")
print(f"Test CSV: {len(test_df)} images")
print(f"Label counts:\n{train_df['label'].value_counts()}")

# Split train into train/val (85/15)
val_df = train_df.sample(frac=0.15, random_state=42)
train_df2 = train_df.drop(val_df.index)

print(f"\nAfter split:")
print(f"Train: {len(train_df2)}, Val: {len(val_df)}, Test: {len(test_df)}")

def copy_images(df, split_dir):
    copied = 0
    skipped = 0
    for _, row in df.iterrows():
        if pd.isna(row["filepath"]):
            skipped += 1
            continue
        src = Path(RAW_DIR) / row["filepath"]
        label = "REAL" if row["label"] == 0 else "FAKE"
        dst_dir = Path(split_dir) / label
        dst = dst_dir / src.name

        if src.exists():
            shutil.copy(src, dst)
            copied += 1
        else:
            skipped += 1

    print(f"  Copied: {copied}, Skipped: {skipped}")

print("\nCopying train images...")
copy_images(train_df2, TRAIN_DIR)

print("Copying val images...")
copy_images(val_df, VAL_DIR)

print("Copying test images...")
copy_images(test_df, TEST_DIR)

print("\nDone! Dataset 2 added successfully.")