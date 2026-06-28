# Occlusion-Resilient Human Tracking and Safety Monitoring
# with Adaptive Logging using an LLM Semantic Layer

**Student:** Jupudi Ravichandra (315502) | M.Eng. Mechatronics, HS Schmalkalden
**Supervisor:** Prof. Ing. Frank Schrodl | **Co-Supervisor:** Y. Strigina, M.Eng.
**Deadline:** August 25, 2026

## The Problem
AMRs lose track of people during occlusion. When a person re-emerges, standard
pipelines reassign a new ID - a safety failure under ISO 10218/15066.

## Dataset
Custom dataset built from real-world AMR deployment trials conducted at a live
facility in Germany - not a controlled lab or simulated indoor environment.
ROS bag files (.bag) were recorded from an Intel RealSense camera mounted on
the robot during actual on-site operations and provided for this thesis.
13 bag files were processed using ROS Noetic: every 3rd frame was extracted,
quality-filtered (blur/dark removal), auto-labeled with YOLOv8s (person class),
and split 70/15/15. Final dataset: 5,773 labeled frames.
Topic: /camera/color/image_raw | Frame step: 3 | JPEG quality: 85

## Three Pillars
1. Occlusion-Resilient Tracking: YOLOv10n + Faster R-CNN verifier + Deep SORT Re-ID
2. Safety Zone Monitoring: Depth Anything V2 (monocular), CRITICAL/WARNING/SAFE zones
3. LLM Semantic Reasoning: Gemma-2-2B (Jetson) / Gemini-2.5-Flash (Colab), blind-spot events only, rule-based fallback

## Pipeline Status
| Phase | Module | Status |
|-------|--------|--------|
| 1 | data_preparation/ | Done - 5,773 frames from real-world ROS bags |
| 2 | yolo_training/ | Done - mAP50=0.929, mAP50-95=0.761 |
| 3 | rcnn_verifier/ | Done - loss 0.0508 (58.2% reduction, 10 epochs) |
| 4 | deepsort_tracking/ | Done - 14.2fps, 6 IDs, 781 records |
| 5 | depth_estimation/ | Done - 8.11fps, 764 detections, ViT-Small |
| 6 | verification_gate/ | Done - 291 pairs, 73 CRITICAL / 115 WARNING / 103 SAFE |
| 7 | llm_semantic_layer/ | Done - Gemini-2.5-Flash, 13/73 LLM calls, BS-1/2/3 taxonomy |
| 8 | evaluation/ | Pending - July (MOTA, IDF1, AbsRel, LLM quality metrics) |
| 9 | ros2_integration/ | Pending - July (Jetson Orin, TensorRT, Ollama Gemma-2-2B) |
