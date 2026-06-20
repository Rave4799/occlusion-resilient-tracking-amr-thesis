# Occlusion-Resilient Human Tracking and Safety Monitoring
# with Adaptive Logging using an LLM Semantic Layer

**Student:** Jupudi Ravichandra (315502) | M.Eng. Mechatronics & Robotics
**Institution:** Hochschule Schmalkalden (HS Schmalkalden)
**Supervisor:** Prof. Ing. Frank Schrodl | **Co-Supervisor:** Y. Strigina, M.Eng.
**Registered:** March 24, 2026 | **Deadline:** August 25, 2026

---

## The Problem

Autonomous Mobile Robots (AMRs) operating in shared human-robot workspaces face a
critical safety challenge: humans disappear. Occlusion behind shelves, machinery,
or other robots causes standard detection pipelines to lose track of people entirely.
When a person re-emerges, the robot has no memory of who they are or where they came
from. In a lab or warehouse setting governed by ISO 10218 and ISO/TS 15066, this is
not just a tracking failure - it is a safety violation.

Existing solutions rely on single-model detection (YOLO alone) or simple re-entry
heuristics that fail under prolonged occlusion. None combine adaptive appearance-based
re-identification with semantic reasoning about blind spots.

---

## The Approach

This thesis proposes a three-pillar hybrid system built on top of a ROS 2 Humble
pipeline, deployed on a Jetson Orin edge device in the HS Schmalkalden robotics lab.

**Pillar 1 - Occlusion-Resilient Tracking**
A hybrid perception pipeline combines YOLOv10 (fast primary detector) with Faster
R-CNN ResNet-50 FPN (async verifier, triggered when YOLO confidence < 0.6 or person
distance < 2.0m). Deep SORT with a MobileNet appearance embedder maintains identity
across occlusion by matching Re-ID feature vectors, not just bounding box overlap.

**Pillar 2 - Safety Zone Monitoring**
Depth Anything V2 estimates metric depth from monocular RGB frames. Three concentric
safety zones are defined per ISO 10218/15066:
  CRITICAL  < 1.0m  - immediate stop
  WARNING   1.0-2.5m - speed reduction
  SAFE      > 2.5m  - normal operation
A 6-trigger verification gate fuses zone events with perception confidence to decide
when to escalate from rule-based to LLM reasoning.

**Pillar 3 - LLM Semantic Safety Reasoning**
Gemma-2-2B runs locally via Ollama as an async daemon thread. It fires ONLY on
blind-spot events: escalating occlusion duration, pre-emergence prediction from
trajectory, and multi-track conflict resolution. It does not describe scenes or
replace perception - it reasons about what the robot cannot see. A deterministic
rule-based fallback ensures safety even when the LLM is slow or unavailable.

---

## Dataset

Collected from 13 ROS bag recordings in the HS Schmalkalden robotics lab.
~45,000 raw frames extracted, filtered to 5,773 high-quality labeled frames.
Auto-labeled using YOLOv8s, manually verified. 1 class: person.
Split: 4,618 train / 1,155 val. Stored on Google Drive + SSD backup.

---

## Pipeline Status

| Phase | Module | Description | Status |
|-------|--------|-------------|--------|
| 1 | data_preparation/ | 13 ROS bags to 5773 labeled frames | Done |
| 2 | yolo_training/ | YOLOv10 fine-tuned on custom dataset | Done |
| 3 | rcnn_verifier/ | Faster R-CNN async verifier, loss 0.0508 | Done |
| 4 | deepsort_tracking/ | Deep SORT + MobileNet Re-ID embedder | In Progress |
| 5 | depth_estimation/ | Depth Anything V2, ISO safety zones | Pending |
| 6 | verification_gate/ | 6-trigger fusion gate | Pending |
| 7 | llm_semantic_layer/ | Gemma-2-2B, blind-spot only reasoning | Pending |
| 8 | evaluation/ | MOTA, IDF1, mAP@0.5, AbsRel | Pending |
| 9 | ros2_integration/ | ROS 2 Humble, Jetson Orin deployment | Pending |

---

## Key Design Decisions

- LLM fires last and only on blind-spot events - perception is primary
- Every module fine-tuned on the custom lab dataset (pretrained alone insufficient)
- Rule-based fallback guarantees safety when LLM is unavailable
- Faster R-CNN is async (does not block YOLO real-time loop)
- All model weights stored on Google Drive, not in git (see models/MODEL_REGISTRY.md)

---

## Tech Stack

PyTorch 2.11 | YOLOv10 | Faster R-CNN ResNet-50 FPN | Deep SORT |
Depth Anything V2 | Gemma-2-2B | Ollama | ROS 2 Humble | Jetson Orin
