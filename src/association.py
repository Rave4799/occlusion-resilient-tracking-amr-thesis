"""
association.py — Track-Depth Association (bbox-direct sampling).

STATUS: VALIDATED. Ported directly from depth_track_annotation.py.
This REPLACES the old IoU-based Verification Gate entirely — no IoU
matching between separate track/depth logs. For each confirmed track,
depth is sampled directly inside that track's own bounding box.

This is a pure function, no state, no safety verdict — see thesis §4.4.
Terminology discipline: "Track-Depth Association", never "Verification
Gate"; "spatial context bin", never "safety zone".

IMPORTANT — confirmed-track contract:
Only CONFIRMED, FRESHLY-MATCHED tracks may reach this module. The
original development notes observed that raw DeepSORT output including
stale/coasted tracks inflates downstream counts ~2.5x GT if not
filtered. This repo's tracking.update_tracks() already enforces that
filter internally (see tracking.py docstring — it returns only tracks
with is_confirmed() and time_since_update == 0), so no separate filter
step is needed here. This module trusts that contract; it does not
re-check confirmation state.

Locked thresholds (main.pdf Table 6.6 — do not retune):
    P25 = 1.06  (near / mid boundary)   — CV=36.5% across frames, less stable
    P65 = 3.61  (mid / far boundary)    — CV=15.4% across frames, more stable
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackWithDepth:
    """A Track (from tracking.py) enriched with spatial context."""
    track_id: int
    bbox: tuple
    avg_depth: float
    depth_bin: str   # "near" | "mid" | "far" | None (None if degenerate bbox)


def compute_track_avg_depth(depth_map: np.ndarray, bbox: tuple) -> float:
    """Average the raw depth map inside a tracker bbox.

    bbox: (x1, y1, x2, y2) in pixel coords.
    Returns NaN for a degenerate bbox (caller should skip/None-bin it).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = depth_map.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return float("nan")

    region = depth_map[y1:y2, x1:x2]
    return float(np.mean(region))


def bin_depth(avg_depth: float, cfg: dict) -> str:
    """Bin a raw avg_depth value into near / mid / far using the locked
    percentile thresholds from config.yaml. This is spatial context
    only — NOT a safety verdict.
    """
    p25 = cfg["depth"]["p25_threshold"]
    p65 = cfg["depth"]["p65_threshold"]

    if avg_depth < p25:
        return "near"
    elif avg_depth < p65:
        return "mid"
    else:
        return "far"


def associate_track_depth(tracks: list, depth_map: np.ndarray, cfg: dict) -> list:
    """For each track, sample the depth map inside its own bbox and
    assign a near/mid/far bin using the locked P25/P65 thresholds.

    tracks: list[tracking.Track] — already confirmed+fresh, per this
        repo's tracking.update_tracks() contract. No re-filtering here.

    Returns list[TrackWithDepth].
    """
    annotated = []
    for track in tracks:
        avg_depth = compute_track_avg_depth(depth_map, track.bbox)
        depth_bin = bin_depth(avg_depth, cfg) if not np.isnan(avg_depth) else None

        annotated.append(TrackWithDepth(
            track_id=track.track_id,
            bbox=track.bbox,
            avg_depth=avg_depth,
            depth_bin=depth_bin,
        ))

    return annotated
