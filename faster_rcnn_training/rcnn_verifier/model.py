"""Phase 3 - Faster R-CNN Model
"""
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(num_classes=2, device=None):
    model=fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_f=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_f,num_classes)
    if device is None: device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); return model,device

def load_checkpoint(path, num_classes=2):
    model,device=build_model(num_classes)
    model.load_state_dict(torch.load(path,map_location=device))
    model.eval(); return model,device
