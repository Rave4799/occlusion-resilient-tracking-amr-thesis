from dotenv import load_dotenv
load_dotenv()

import cv2
from pathlib import Path
from src.config import load_config
from src import detection, tracking, depth, association, reasoning

cfg = load_config()
print("Loading all models...")
yolo_model, frcnn_model, device = detection.load_models(cfg)
tracker = tracking.init_tracker(cfg)
depth_model = depth.load_depth_model(cfg)

frames_dir = Path("/Volumes/RAD/pipeline_work/test6_clip")
frame_files = sorted(frames_dir.glob("*.jpg"))
print(f"Running full chain on {len(frame_files)} frames...\n")

track_history = {}
events_fired = []

for i, fpath in enumerate(frame_files):
    frame = cv2.imread(str(fpath))

    dets = detection.detect_frame(frame, yolo_model, frcnn_model, device, cfg)
    tracks = tracking.update_tracks(tracker, dets, frame)
    depth_map = depth.run_depth(frame, depth_model)
    tracks_with_depth = association.associate_track_depth(tracks, depth_map, cfg)

    track_history = reasoning.update_track_history(track_history, tracks_with_depth, i)
    event = reasoning.bs1_trigger(track_history, tracks_with_depth, i, cfg)

    if event:
        entry = reasoning.rule_template(event)
        entry = reasoning.llm_enrichment(entry, cfg)
        events_fired.append((i, entry))
        print(f"*** BS-1 FIRED at frame {i} ***")
        print(f"    track_id={entry['track_id']}, frames_stable_before_loss={entry['frames_stable_before_loss']}, frames_lost={entry['frames_lost']}")
        print(f"    last_depth_bin={entry['last_depth_bin']}, last_avg_depth={entry['last_avg_depth']:.3f}")
        print(f"    reasoning: {entry['reasoning']}")
        print(f"    llm_enrichment: {entry['llm_enrichment']}")
        print()

    if i % 30 == 0:
        print(f"[progress] frame {i}/{len(frame_files)}")

print(f"\n=== SUMMARY ===")
print(f"Total frames: {len(frame_files)}")
print(f"BS-1 events fired: {len(events_fired)}")
for i, entry in events_fired:
    print(f"  Frame {i}: track {entry['track_id']}, {entry['frames_lost']} frames lost")
