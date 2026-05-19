# Occlusion-Resilient Tracking and Implementation of Event-Triggered LLM Semantic Layer in Autonomous Mobile Robot

## Project Information

- **Student:** Jupudi Ravichandra (315502)
- **University:** Hochschule Schmalkalden
- **Supervisors:** Prof. Frank Schrödel and Y. Strigina
- **Degree Program:** Master Thesis
- **Start Date:** May 2023

## Overview

This thesis develops an integrated pipeline for robust human tracking in occluded environments and implements an event-triggered Large Language Model (LLM) semantic layer for autonomous mobile robots. The system combines multiple computer vision and deep learning techniques to achieve occlusion-resilient tracking with real-time LLM-based semantic understanding.

### Objectives

- Achieve mAP50 > 0.90 for person detection
- Maintain MOTA > 0.80 and IDF1 > 0.85 for tracking
- Implement depth-aware occlusion handling with AbsRel < 0.15
- Integrate event-triggered LLM reasoning for semantic scene understanding
- Deploy on ROS 2 Humble for mobile robotics platforms

## Project Architecture

### 9-Phase Pipeline

1. **Phase 1: Data Preparation** — Extract RGB frames from ROS bags, create YOLO-format annotations
2. **Phase 2: YOLO Training** — Train YOLOv8 detector on 10,000 frames (70/15/15 split)
3. **Phase 3: RCNN Verification** — Validate detections with Faster R-CNN baseline
4. **Phase 4: DeepSORT Tracking** — Implement multi-object tracking with appearance features
5. **Phase 5: Depth Estimation** — Monocular depth estimation for occlusion reasoning
6. **Phase 6: Verification Gate** — Confidence-based prediction filtering
7. **Phase 7: LLM Semantic Layer** — Event-triggered reasoning for scene semantics
8. **Phase 8: ROS 2 Integration** — Package pipeline for autonomous mobile robots
9. **Phase 9: Evaluation** — Benchmark against MOT metrics and deployment testing

## Folder Structure

```
thesis_ws/
├── data_preparation/          # Phase 1: ROS bag extraction and dataset creation
├── yolo_training/             # Phase 2: YOLOv8 model training and validation
├── rcnn_verifier/             # Phase 3: Faster R-CNN baseline verification
├── deepsort_tracking/         # Phase 4: Multi-object tracking implementation
├── depth_estimation/          # Phase 5: Monocular depth estimation models
├── verification_gate/         # Phase 6: Confidence filtering and gating logic
├── llm_semantic_layer/        # Phase 7: LLM-based semantic reasoning
├── evaluation/                # Phase 9: Metrics computation and benchmarking
├── ros2_integration/          # Phase 8: ROS 2 package structure and nodes
├── configs/                   # Configuration files (YAML, JSON)
├── results/                   # Output logs, metrics, trained weights
├── dataset/                   # Training data (ignored in git, see .gitignore)
├── weights/                   # Pre-trained and trained model weights
├── logs/                      # Execution logs and checkpoints
└── README.md                  # This file
```

## Requirements

### System Requirements
- **OS:** Ubuntu 20.04 LTS
- **CPU:** 8+ cores recommended
- **GPU:** NVIDIA GPU with CUDA 11.0+ (for efficient training)
- **RAM:** 16+ GB recommended

### Software Dependencies

```
Python 3.8
ROS Noetic
ROS 2 Humble
PyTorch 1.13+
torchvision 0.14+
Ultralytics YOLOv8
Faster R-CNN (torchvision)
DeepSORT
MiDaS (depth estimation)
Ollama or OpenAI API (LLM backend)
```

### Python Packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy scipy scikit-learn pandas
pip install rospy rosnumpy message-filters
pip install fastapi pydantic requests
pip install python-rosdep
```

### Installation

1. **Install ROS Noetic** (for data collection)
   ```bash
   sudo apt install ros-noetic-desktop-full
   ```

2. **Install ROS 2 Humble** (for deployment)
   ```bash
   sudo apt install ros-humble-desktop
   ```

3. **Clone and Setup**
   ```bash
   cd ~/thesis_ws
   source /opt/ros/noetic/setup.bash
   source /opt/ros/humble/setup.bash
   ```

## Dataset

- **Source:** ROS bag files from `/media/rad4799/RAD/ROS BAG Files/`
- **Bags:** 2023_05_05, 2023-05-12, 2023-05-13 (13 total)
- **Annotation Format:** YOLO (normalized: class_id, cx, cy, w, h)
- **Single Class:** Person (class_id = 0)
- **Split:** 70% training, 15% validation, 15% test
- **Frame Extraction:** Every 3rd frame (step=3)
- **Target Size:** 10,000 frames minimum

## Metric Targets

| Metric | Target |
|--------|--------|
| mAP50 (Detection) | > 0.90 |
| MOTA (Tracking) | > 0.80 |
| IDF1 (ID F1-Score) | > 0.85 |
| AbsRel (Depth) | < 0.15 |

## Coding Guidelines

- **Language:** Python only
- **Version:** Python 3.8+
- **Path Handling:** Use `pathlib.Path` for all file operations
- **Base Directory:** `BASE_DIR = Path(__file__).resolve().parent.parent`
- **Comments:** Inline comments on all meaningful lines
- **Functions:** Modular design with single responsibility
- **Progress:** Print statements during long operations
- **No Hardcoding:** Avoid hardcoded paths, usernames, or home directories

## Running the Pipeline

Each phase has standalone execution:

```bash
# Phase 1: Data Preparation
python data_preparation/extract_frames.py

# Phase 2: Train YOLO
python yolo_training/train.py

# Phase 4: Run Tracking
python deepsort_tracking/track.py

# Phase 7: LLM Reasoning
python llm_semantic_layer/reasoning.py

# Phase 9: Evaluation
python evaluation/compute_metrics.py
```

## Results and Checkpoints

- **Models:** Saved in `weights/`
- **Logs:** Training logs in `logs/`
- **Metrics:** Evaluation results in `results/`
- **Outputs:** Tracked sequences and predictions in `results/`

## Git Configuration

All folders include `.gitkeep` to maintain structure. The following are excluded:

```
dataset/          (large training data)
weights/          (large model files)
logs/             (runtime outputs)
results/          (evaluation outputs)
*.bag             (ROS bag files)
```

## References

- YOLOv8: https://github.com/ultralytics/ultralytics
- DeepSORT: https://github.com/nwojke/deep_sort
- MiDaS: https://github.com/isl-org/MiDaS
- ROS 2 Documentation: https://docs.ros.org/en/humble/
- PyTorch: https://pytorch.org/

## Contact

For questions or clarifications, contact the thesis supervisor at Hochschule Schmalkalden.

---

**Last Updated:** May 19, 2026