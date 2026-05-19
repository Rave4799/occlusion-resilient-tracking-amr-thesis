from pathlib import Path
import shutil
import random

# Base directory for workspace
BASE_DIR = Path(__file__).resolve().parent.parent

# Source and destination
src = BASE_DIR / 'dataset' / 'train' / 'images_filtered'
dst = BASE_DIR / 'dataset' / 'train' / 'images_selected'
dst.mkdir(parents=True, exist_ok=True)

# Get all filtered images
all_images = sorted([p.name for p in src.iterdir() if p.is_file()])
print(f"Total filtered images: {len(all_images)}")

# Sample every 3rd frame
sampled = all_images[::3]
print(f"After every-3rd sampling: {len(sampled)}")

# Cap at 10000
if len(sampled) > 10000:
    random.seed(42)
    sampled = random.sample(sampled, 10000)
    
print(f"Final selection: {len(sampled)}")

# Copy selected frames
for i, fname in enumerate(sampled):
    shutil.copy2(src / fname, dst / fname)
    if i % 1000 == 0:
        print(f"Copied {i}/{len(sampled)}")

print(f"Done. {len(sampled)} frames saved to {dst}")