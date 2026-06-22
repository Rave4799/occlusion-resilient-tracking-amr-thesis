# Occlusion-Resilient Human Tracking and Safety Monitoring
# with Adaptive Logging using an LLM Semantic Layer

**Student:** Jupudi Ravichandra (315502) | M.Eng. Mechatronics, HS Schmalkalden
**Supervisor:** Prof. Ing. Frank Schrodl | **Co-Supervisor:** Y. Strigina, M.Eng.
**Deadline:** August 25, 2026

## The Problem
AMRs lose track of people during occlusion. When a person re-emerges, standard
pipelines reassign a new ID - a safety failure under ISO 10218/15066.

## Three Pillars
1. Occlusion-Resilient Tracking: YOLOv10 + Faster R-CNN verifier + Deep SORT Re-ID
2. Safety Zone Monitoring: Depth Anything V2, CRITICAL/WARNING/SAFE zones
3. LLM Semantic Reasoning: Gemma-2-2B, blind-spot events only, rule-based fallback

## Pipeline Status
| Phase | Module | Status |
|-------|--------|--------|
| 1 | data_preparation/ | Done |
| 2 | yolo_training/ | Done |
| 3 | rcnn_verifier/ | Done - loss 0.0508 |
| 4 | deepsort_tracking/ | In Progress |
| 5 | depth_estimation/ | Pending |
| 6 | verification_gate/ | Pending |
| 7 | llm_semantic_layer/ | Pending |
| 8 | evaluation/ | Pending |
| 9 | ros2_integration/ | Pending |
