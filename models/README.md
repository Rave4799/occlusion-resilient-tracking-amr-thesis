# Trained Model Weights

This folder contains trained model weights for the thesis pipeline. **These files are not tracked in git due to large file sizes** (typically 100MB-500MB each).

## Important Notes

- All `.pt`, `.pth`, and `.t7` files are excluded from git (see `.gitignore`)
- Models must be downloaded or trained locally before running the pipeline
- Ensure sufficient disk space (~2GB) for all models
- Use the download links or training commands provided below

---

## Models

### 1. YOLOv10s Fine-tuned (yolov10s_finetuned.pt)

**Purpose:** Person detection on custom dataset  
**Framework:** Ultralytics YOLOv10  
**Size:** ~100-150 MB  
**Input:** RGB images (640x640)  
**Output:** Bounding boxes with confidence scores

#### Download

```bash
# Download from Hugging Face (if available)
wget https://huggingface.co/[username]/yolov10s_finetuned/resolve/main/yolov10s_finetuned.pt -O models/yolov10s_finetuned.pt

# Or download from custom drive link
# [Add your custom download link here]
```

#### Training Command

```bash
cd yolo_training

# Train YOLOv10s on custom dataset
python train.py \
  --model yolov10s.pt \
  --data dataset_config.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch-size 16 \
  --device 0 \
  --name yolov10s_finetuned

# Copy trained weights to models/
cp runs/detect/yolov10s_finetuned/weights/best.pt ../models/yolov10s_finetuned.pt
```

---

### 2. Faster R-CNN Verifier (fasterrcnn_verifier.pth)

**Purpose:** Baseline verification of YOLOv10 detections  
**Framework:** PyTorch Faster R-CNN (torchvision)  
**Size:** ~150-200 MB  
**Input:** RGB images (800x800 minimum)  
**Output:** Verified bounding boxes and classification scores

#### Download

```bash
# Pretrained Faster R-CNN on COCO
# (Fine-tune on custom dataset or use pretrained)
wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth -O models/fasterrcnn_verifier.pth
```

#### Training Command

```bash
cd rcnn_verifier

# Fine-tune Faster R-CNN on custom dataset
python train.py \
  --model fasterrcnn_resnet50_fpn \
  --dataset dataset_config.json \
  --epochs 50 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --device cuda:0

# Save trained weights
cp checkpoints/best.pth ../models/fasterrcnn_verifier.pth
```

---

### 3. Depth Anything V2 ViT-B (depth_anything_v2_vitb.pth)

**Purpose:** Monocular depth estimation for occlusion reasoning  
**Framework:** MiDaS / Depth Anything V2  
**Size:** ~350-400 MB  
**Input:** RGB images (any resolution)  
**Output:** Depth maps (normalized 0-1 or 0-255)

#### Download

```bash
# Download from official Depth Anything V2 repository
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything-V2/resolve/main/checkpoints/depth_anything_v2_vitb.pth -O models/depth_anything_v2_vitb.pth

# Alternative: Use model hub
python -c "
from depth_anything_v2.dpt import DepthAnythingV2
model = DepthAnythingV2(encoder='vitb', max_depth=80, load_pretrained=True)
model.save_pretrained('models/depth_anything_v2_vitb.pth')
"
```

#### Training Command

```bash
cd depth_estimation

# Train depth estimation model (if using custom dataset)
python train.py \
  --model depth_anything_v2 \
  --backbone vitb \
  --dataset custom_depth_dataset.json \
  --epochs 100 \
  --batch-size 32

# Save checkpoint
cp checkpoints/final.pth ../models/depth_anything_v2_vitb.pth
```

---

### 4. Deep SORT Re-ID Embedder (ckpt.t7)

**Purpose:** Person re-identification feature extraction for tracking  
**Framework:** PyTorch (Market-1501 / DukeMTMC pretrained)  
**Size:** ~25-50 MB  
**Input:** Person crops (128x64 or 256x128)  
**Output:** 128-dim feature embeddings

#### Download

```bash
# Download Deep SORT model from official repository
wget https://github.com/nwojke/deep_sort_pytorch/releases/download/ckpt/ckpt.t7 -O models/ckpt.t7

# Alternative: From custom hosted location
wget https://example.com/models/ckpt.t7 -O models/ckpt.t7
```

#### Training Command

```bash
cd deepsort_tracking

# Train Re-ID embedder on market dataset or custom data
python train_reid.py \
  --dataset market1501 \
  --model resnet50 \
  --epochs 120 \
  --batch-size 64 \
  --learning-rate 0.001

# Convert to .t7 format
python -c "
import torch
model = torch.load('checkpoints/reid_model.pth')
torch.jit.script(model).save('models/ckpt.t7')
"
```

---

## Setup Instructions

1. **Create models folder** (already exists)
   ```bash
   mkdir -p models
   ```

2. **Download all models** using links above
   ```bash
   cd models
   # Run wget commands for each model
   ```

3. **Verify model integrity**
   ```bash
   python -c "
   import os
   models = [
       'yolov10s_finetuned.pt',
       'fasterrcnn_verifier.pth',
       'depth_anything_v2_vitb.pth',
       'ckpt.t7'
   ]
   for m in models:
       if os.path.exists(m):
           size_mb = os.path.getsize(m) / (1024**2)
           print(f'✓ {m}: {size_mb:.1f} MB')
       else:
           print(f'✗ {m}: NOT FOUND')
   "
   ```

4. **Run pipeline** (models will be auto-loaded)
   ```bash
   python data_preparation/extract_frames.py
   python yolo_training/train.py
   python deepsort_tracking/track.py
   ```

---

## Troubleshooting

### Models not found
```bash
# Check if models exist
ls -lah models/

# Re-download if missing
wget [download_link] -O models/[model_name]
```

### Out of memory during inference
- Reduce batch size in scripts
- Use smaller model variants (YOLOv10n instead of YOLOv10s)
- Enable half-precision (FP16) mode

### Model compatibility issues
- Verify PyTorch version matches training environment
- Check CUDA version compatibility
- Ensure torchvision, ultralytics, and opencv versions are correct

---

## References

- YOLOv10: https://github.com/THU-MIG/yolov10
- Faster R-CNN: https://pytorch.org/vision/stable/models.html#faster-rcnn
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- Deep SORT: https://github.com/nwojke/deep_sort_pytorch

---

**Last Updated:** May 19, 2026
