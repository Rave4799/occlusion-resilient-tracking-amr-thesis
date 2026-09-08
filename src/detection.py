"""
detection.py — Dual-detector: YOLOv10n (primary) + Faster R-CNN (verifier).

STATUS: VALIDATED. Ported directly from Tracker_GT_Validation.ipynb —
this is the exact code that produced the thesis's real MOTA/IDF1 numbers
and the tracker_yolo_only.mp4 / tracker_dual_detector.mp4 demo videos.

IMPORTANT — color space: YOLO consumes raw BGR frames (as read by
cv2.imread/VideoCapture) directly. Faster R-CNN requires RGB — the BGR
frame is converted before tensor conversion. Getting this backwards is
a silent bug (models still run, boxes are just wrong), so it's called
out explicitly at each call site.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO


@dataclass
class Detection:
    """A single detection: bounding box + confidence + source detector."""
    bbox: tuple       # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    source: str       # "yolo" or "frcnn"


def load_models(cfg: dict):
    """Load YOLOv10n and Faster R-CNN weights per config paths.

    Returns: (yolo_model, frcnn_model, device)
    """
    device = torch.device(
        cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )

    yolo_weights = cfg["paths"]["yolo_weights"]
    yolo_model = YOLO(str(yolo_weights))

    frcnn_weights = cfg["paths"]["frcnn_weights"]
    ckpt = torch.load(str(frcnn_weights), map_location="cpu")
    # Some checkpoints wrap the state dict; handle both forms.
    state = ckpt if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt["model_state_dict"]

    frcnn_model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
    frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # background + person
    frcnn_model.load_state_dict(state)
    frcnn_model = frcnn_model.to(device).eval()

    return yolo_model, frcnn_model, device


def run_yolo(frame_bgr, model, conf_threshold: float) -> list:
    """Run YOLOv10n on a single BGR frame. Returns list[Detection].

    NOTE: frame is passed as raw BGR — YOLO/ultralytics handles this
    internally, do NOT convert to RGB here.
    """
    results = model(frame_bgr, conf=conf_threshold, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    return [
        Detection(bbox=tuple(box), confidence=float(conf), source="yolo")
        for box, conf in zip(boxes, confs)
    ]


def run_frcnn(frame_bgr, model, device, conf_threshold: float) -> list:
    """Run Faster R-CNN on a single frame. Returns list[Detection].

    NOTE: Faster R-CNN requires RGB input — the BGR frame is converted
    here before tensor conversion. This conversion is REQUIRED; skipping
    it does not error, it just silently produces wrong boxes.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = TF.to_tensor(frame_rgb).to(device)

    with torch.no_grad():
        output = model([img_tensor])[0]

    # label == 1 is "person" (0 is background, per the 2-class head)
    mask = (output["labels"] == 1) & (output["scores"] > conf_threshold)
    boxes = output["boxes"][mask].cpu().numpy()
    scores = output["scores"][mask].cpu().numpy()

    return [
        Detection(bbox=tuple(box), confidence=float(score), source="frcnn")
        for box, score in zip(boxes, scores)
    ]


def _compute_iou(a: tuple, b: tuple) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fuse_detections(yolo_dets: list, frcnn_dets: list, iou_threshold: float) -> list:
    """Fuse YOLO + Faster R-CNN detections by IoU overlap (dual-detector
    always-on ablation, thesis §5.4.2 — NOT the selective trigger of §4.2).

    Every YOLO detection is kept. An FRCNN detection is added only if it
    does NOT overlap (IoU > threshold) any existing YOLO detection —
    i.e. FRCNN only contributes detections YOLO missed.
    """
    fused = list(yolo_dets)
    for fd in frcnn_dets:
        if not any(_compute_iou(fd.bbox, yd.bbox) > iou_threshold for yd in yolo_dets):
            fused.append(fd)
    return fused


def detect_frame(frame_bgr, yolo_model, frcnn_model, device, cfg: dict) -> list:
    """Full detection stage for one frame: run both detectors, fuse results.
    This is the function run_pipeline.py calls per frame.
    """
    yolo_dets = run_yolo(frame_bgr, yolo_model, cfg["detection"]["yolo_conf_threshold"])
    frcnn_dets = run_frcnn(frame_bgr, frcnn_model, device, cfg["detection"]["frcnn_conf_threshold"])
    return fuse_detections(yolo_dets, frcnn_dets, cfg["detection"]["fusion_iou_threshold"])
