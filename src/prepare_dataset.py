import os
import shutil
import random
from pathlib import Path

# Paths
RAW_TRAIN_REAL = "data/raw/train/REAL"
RAW_TRAIN_FAKE = "data/raw/train/FAKE"
RAW_TEST_REAL = "data/raw/test/REAL"
RAW_TEST_FAKE = "data/raw/test/FAKE"

# Output paths
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# Create output folders
for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(f"{split}/REAL", exist_ok=True)
    os.makedirs(f"{split}/FAKE", exist_ok=True)

def split_and_copy(source_dir, label, val_ratio=0.15):
    files = list(Path(source_dir).glob("*.jpg")) + \
            list(Path(source_dir).glob("*.png"))
    random.shuffle(files)
    
    val_count = int(len(files) * val_ratio)
    val_files = files[:val_count]
    train_files = files[val_count:]
    
    for f in train_files:
        shutil.copy(f, f"{TRAIN_DIR}/{label}/{f.name}")
    for f in val_files:
        shutil.copy(f, f"{VAL_DIR}/{label}/{f.name}")
    
    print(f"{label} - Train: {len(train_files)}, Val: {len(val_files)}")

def copy_test(source_dir, label):
    files = list(Path(source_dir).glob("*.jpg")) + \
            list(Path(source_dir).glob("*.png"))
    for f in files:
        shutil.copy(f, f"{TEST_DIR}/{label}/{f.name}")
    print(f"{label} - Test: {len(files)}")

print("Organizing dataset...")
random.seed(42)

split_and_copy(RAW_TRAIN_REAL, "REAL")
split_and_copy(RAW_TRAIN_FAKE, "FAKE")
copy_test(RAW_TEST_REAL, "REAL")
copy_test(RAW_TEST_FAKE, "FAKE")

print("Done! Dataset organized into train/val/test splits.")