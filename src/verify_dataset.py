import os
from pathlib import Path
from PIL import Image
import cv2

# Dataset splits and labels to check
SPLITS = {
    "train": "data/train",
    "val": "data/val",
    "test": "data/test"
}
LABELS = ["REAL", "FAKE"]

def verify_image(filepath):
    """Try to open image with both PIL and OpenCV."""
    try:
        img = Image.open(filepath)
        img.verify()
        img = cv2.imread(str(filepath))
        if img is None:
            return False, "OpenCV could not read file"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def verify_split(split_name, split_path):
    print(f"\nChecking {split_name}...")
    total = 0
    corrupted = []
    size_counts = {}

    for label in LABELS:
        folder = Path(split_path) / label
        files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        
        for f in files:
            total += 1
            ok, msg = verify_image(f)
            
            if not ok:
                corrupted.append((f, msg))
            else:
                # Track image sizes
                img = cv2.imread(str(f))
                h, w = img.shape[:2]
                size = f"{w}x{h}"
                size_counts[size] = size_counts.get(size, 0) + 1

        print(f"  {label}: {len(files)} images")

    print(f"  Total: {total} images")
    print(f"  Corrupted: {len(corrupted)}")
    
    if corrupted:
        print("  Corrupted files:")
        for f, msg in corrupted[:10]:
            print(f"    {f}: {msg}")
    
    print(f"  Image sizes found: {dict(list(size_counts.items())[:5])}")
    return corrupted

print("Starting dataset verification...")
all_corrupted = []

for split_name, split_path in SPLITS.items():
    corrupted = verify_split(split_name, split_path)
    all_corrupted.extend(corrupted)

print(f"\nVerification complete!")
print(f"Total corrupted images: {len(all_corrupted)}")

if len(all_corrupted) == 0:
    print("All images are valid and ready for training!")