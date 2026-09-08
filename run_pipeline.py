#!/usr/bin/env python3
"""
run_pipeline.py — single entry point chaining all five pipeline stages.

    detection -> tracking -> depth -> association -> reasoning

STATUS: COMPLETE. This is the exact five-stage chain validated on both
demo clips (Test6: 2 BS-1 events; occlusion clip: 5 BS-1 events, see
Known_Issues_Defense_Notes.pdf for full validation history and caveats).

Usage:
    python run_pipeline.py --input data/demo_clip.mp4 --output outputs/

Two demo clips, two different claims (see Known_Issues_Defense_Notes.pdf
D1 for the full explanation — do not conflate them):
    1. Test6 clip        -> confirms BS-1 fires correctly on a KNOWN
                             occlusion event (functional correctness,
                             training-adjacent data)
    2. Untouched-bag clip -> demonstrates the full pipeline generalizing
                             to genuinely unseen data (the real
                             generalization evidence)

Timing reference (validated, macOS CPU-only, no GPU): ~1.9s/frame,
~0.52 FPS. This is CPU inference across 3 models per frame (YOLO,
Faster R-CNN, DepthAnythingV2) — GPU would be substantially faster.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

from src.config import load_config
from src import detection, tracking, depth, association, reasoning

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[pipeline] python-dotenv not installed — GEMINI_API_KEY must be "
          "set some other way (shell export, etc.) for LLM enrichment to work. "
          "Rule-based reasoning is unaffected either way.")


BIN_COLORS = {
    "near": (0, 0, 255),
    "mid": (0, 165, 255),
    "far": (0, 200, 0),
    None: (128, 128, 128),
}


def _check_inputs_exist(cfg: dict, input_path: Path):
    """Fail early and clearly if model weights or the input path are
    missing, instead of letting a cryptic exception surface later mid-run
    (e.g. after weights have already started loading).
    """
    problems = []

    yolo_weights = cfg["paths"]["yolo_weights"]
    if not yolo_weights.exists():
        problems.append(
            f"YOLO weights not found at: {yolo_weights}\n"
            f"    Check config.yaml's paths.yolo_weights, or place the file there."
        )

    frcnn_weights = cfg["paths"]["frcnn_weights"]
    if not frcnn_weights.exists():
        problems.append(
            f"Faster R-CNN weights not found at: {frcnn_weights}\n"
            f"    Check config.yaml's paths.frcnn_weights, or place the file there."
        )

    if not input_path.exists():
        problems.append(
            f"Input path does not exist: {input_path}\n"
            f"    Provide a valid video file or a directory of frame images."
        )
    elif input_path.is_dir():
        has_frames = any(input_path.glob("*.jpg")) or any(input_path.glob("*.png"))
        if not has_frames:
            problems.append(
                f"Input directory has no .jpg or .png files: {input_path}"
            )

    if problems:
        print("\n[pipeline] STARTUP CHECK FAILED — cannot proceed:\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}\n")
        sys.exit(1)


def process_clip(input_path: Path, output_dir: Path, cfg: dict, clip_label: str = ""):
    """Run the full five-stage pipeline on every frame of a video or
    image-frame directory, writing an annotated output video and a
    BS-1 event log CSV.
    """
    _check_inputs_exist(cfg, input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[pipeline] loading models...")
    try:
        yolo_model, frcnn_model, device = detection.load_models(cfg)
        tracker = tracking.init_tracker(cfg)
        depth_model = depth.load_depth_model(cfg)
    except Exception as e:
        print(f"\n[pipeline] FATAL — model loading failed: {type(e).__name__}: {e}")
        print("[pipeline] Check that all packages in requirements.txt are "
              "installed and model weight files are valid, uncorrupted files.")
        sys.exit(1)

    if input_path.is_dir():
        frame_paths = sorted(input_path.glob("*.jpg")) or sorted(input_path.glob("*.png"))
        frame_source = iter(cv2.imread(str(p)) for p in frame_paths)
        total_frames = len(frame_paths)
        cap = None
    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"\n[pipeline] FATAL — could not open video file: {input_path}")
            print("[pipeline] Check the file is a valid, uncorrupted video "
                  "in a format OpenCV supports (mp4/avi/mov).")
            sys.exit(1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        def _video_frames():
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                yield f
        frame_source = _video_frames()

    print(f"[pipeline] processing {total_frames} frames from {input_path}")

    track_history = {}
    events_log = []
    writer = None
    frames_processed = 0
    frames_skipped = 0

    for frame_idx, frame in enumerate(frame_source):
        if frame is None:
            frames_skipped += 1
            print(f"[pipeline] WARNING — frame {frame_idx} failed to load "
                  f"(corrupt file or read error), skipping.")
            continue

        try:
            detections = detection.detect_frame(frame, yolo_model, frcnn_model, device, cfg)
            tracks = tracking.update_tracks(tracker, detections, frame)
            depth_map = depth.run_depth(frame, depth_model)
            tracks_with_depth = association.associate_track_depth(tracks, depth_map, cfg)

            track_history = reasoning.update_track_history(track_history, tracks_with_depth, frame_idx)
            event = reasoning.bs1_trigger(track_history, tracks_with_depth, frame_idx, cfg)

            if event is not None:
                entry = reasoning.rule_template(event)
                entry = reasoning.llm_enrichment(entry, cfg)
                events_log.append(entry)
                print(f"[frame {frame_idx}] *** BS-1 fired *** track {event.track_id}, "
                      f"stable={event.frames_stable_before_loss}, lost={event.frames_lost}")

        except Exception as e:
            # A single bad frame should not kill an otherwise-working run.
            # Log it clearly and continue — this matters for a long real
            # deployment run where one corrupt/unusual frame shouldn't
            # take down hours of otherwise-good output.
            frames_skipped += 1
            print(f"[pipeline] WARNING — frame {frame_idx} raised "
                  f"{type(e).__name__}: {e}. Skipping this frame, continuing.")
            tracks_with_depth = []
            event = None

        annotated = frame.copy()
        for t in tracks_with_depth:
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            color = BIN_COLORS.get(t.depth_bin, BIN_COLORS[None])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{t.track_id} [{t.depth_bin or '?'}]"
            cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if event is not None:
            cv2.putText(annotated, "BS-1 EVENT", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        if clip_label:
            cv2.putText(annotated, clip_label, (10, annotated.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer is None:
            h, w = annotated.shape[:2]
            out_video_path = output_dir / "annotated_output.mp4"
            writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))

        writer.write(annotated)
        frames_processed += 1

        if frame_idx % 25 == 0:
            print(f"[pipeline] frame {frame_idx}/{total_frames}")

    if writer is not None:
        writer.release()
    if cap is not None:
        cap.release()

    log_path = output_dir / "bs1_events.csv"
    with open(log_path, "w", newline="") as f:
        if events_log:
            writer_csv = csv.DictWriter(f, fieldnames=list(events_log[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(events_log)
        else:
            f.write("event_type,track_id,trigger_frame,last_seen_frame,"
                    "frames_stable_before_loss,frames_lost,last_depth_bin,"
                    "last_avg_depth,reasoning,llm_enrichment\n")

    print(f"\n[pipeline] done. {len(events_log)} BS-1 event(s) across "
          f"{frames_processed} processed frames "
          f"({frames_skipped} skipped due to errors)." if frames_skipped else
          f"\n[pipeline] done. {len(events_log)} BS-1 event(s) across "
          f"{frames_processed} frames.")
    print(f"[pipeline] annotated video: {output_dir / 'annotated_output.mp4'}")
    print(f"[pipeline] event log: {log_path}")

    return events_log


def main():
    parser = argparse.ArgumentParser(
        description="Run the full occlusion-resilient tracking pipeline end-to-end."
    )
    parser.add_argument("--input", type=str, required=True,
                         help="Path to input video file OR a directory of frame images.")
    parser.add_argument("--output", type=str, default=None,
                         help="Output directory. Defaults to config.yaml's paths.output_dir.")
    parser.add_argument("--label", type=str, default="",
                         help="Short label overlaid on the output video (e.g. 'Test6').")
    args = parser.parse_args()

    try:
        cfg = load_config()
    except Exception as e:
        print(f"\n[pipeline] FATAL — could not load config.yaml: {type(e).__name__}: {e}")
        print("[pipeline] Check config.yaml exists in the repo root and is valid YAML.")
        sys.exit(1)

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else cfg["paths"]["output_dir"]

    process_clip(input_path, output_dir, cfg, clip_label=args.label)


if __name__ == "__main__":
    main()
