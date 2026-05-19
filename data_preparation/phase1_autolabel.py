from pathlib import Path
import cv2
from ultralytics import YOLO

# Base directory for workspace
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
images_dir = BASE_DIR / 'dataset' / 'train' / 'images_selected'
labels_dir = BASE_DIR / 'dataset' / 'train' / 'labels'
model_path = BASE_DIR / 'weights' / 'yolov8s.pt'
labels_dir.mkdir(parents=True, exist_ok=True)

# Load pretrained YOLO model
model = YOLO(str(model_path))

# Get all images
images = sorted([p.name for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
print(f"Total images to label: {len(images)}")

labeled = 0
for i, fname in enumerate(images):
    img_path = images_dir / fname
    results = model(str(img_path), conf=0.3, classes=[0], verbose=False)[0]
    boxes = results.boxes

    if len(boxes) == 0:
        continue

    label_file = labels_dir / (fname.rsplit('.', 1)[0] + '.txt')
    h, w = results.orig_shape

    with open(label_file, "w") as f:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    labeled += 1

    if i % 100 == 0:
        print(f"Progress: {i}/{len(images)} | labeled: {labeled}")

print(f"Done. {labeled}/{len(images)} images labeled.")
