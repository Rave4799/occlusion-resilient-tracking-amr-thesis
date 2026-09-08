"""
tracking.py — DeepSORT + MobileNet Re-ID identity-preserving tracker.

STATUS: VALIDATED. Ported directly from Tracker_GT_Validation.ipynb — the
exact config that produced the thesis's modular comparison (DeepSORT 3 IDs
vs. ByteTrack 17 vs. OC-SORT 68 on identical detection input) and the
tracker_yolo_only.mp4 / tracker_dual_detector.mp4 demo videos.

CONFIG NOTE: the real, validated values are max_age=30, n_init=3,
max_cosine_distance=0.4, nn_budget=100 — confirmed directly from the
notebook. Do not retune these without re-running validation.
"""
from dataclasses import dataclass

from deep_sort_realtime.deepsort_tracker import DeepSort


@dataclass
class Track:
    """A single tracked person at the current frame."""
    track_id: int
    bbox: tuple        # (x1, y1, x2, y2)


def init_tracker(cfg: dict):
    """Initialize DeepSORT with MobileNet Re-ID embedder, using the
    validated config from config.yaml.
    """
    return DeepSort(
        max_age=cfg["tracking"]["max_age"],
        n_init=cfg["tracking"]["n_init"],
        max_cosine_distance=cfg["tracking"]["max_cosine_distance"],
        nn_budget=cfg["tracking"]["nn_budget"],
    )


def update_tracks(tracker, detections: list, frame_rgb_or_bgr) -> list:
    """Feed this frame's fused detections into the tracker, return the
    current list[Track] with track IDs assigned/maintained.

    NOTE on frame color space: deep-sort-realtime's internal embedder
    only uses the frame for appearance-feature extraction inside each
    detection's bbox — pass frame_rgb when following the dual-detector
    path (matches the validated notebook, which converts to RGB for
    Faster R-CNN and reuses that same RGB frame for the tracker update).
    For the YOLO-only path, the validated notebook passes the raw BGR
    frame directly — the two paths are NOT required to match each other,
    just to match what was validated for that specific configuration.

    Only CONFIRMED tracks with a fresh detection match this frame are
    returned (time_since_update == 0) — this is the "no-coasting" filter
    used throughout thesis validation, and affects MOTA/IDF1 interpretation
    (see thesis Ch.6 notes on ID-switch counting under this filter).

    detections: list[detection.Detection] — converted here to the
    ([x, y, w, h], confidence, class) tuple format deep-sort-realtime expects.
    """
    dets_formatted = [
        (
            [det.bbox[0], det.bbox[1], det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]],
            det.confidence,
            0,  # single class: person
        )
        for det in detections
    ]

    raw_tracks = tracker.update_tracks(dets_formatted, frame=frame_rgb_or_bgr)

    confirmed = []
    for t in raw_tracks:
        if not t.is_confirmed() or t.time_since_update > 0:
            continue
        l, top, r, b = t.to_ltrb()
        confirmed.append(Track(track_id=t.track_id, bbox=(l, top, r, b)))

    return confirmed
