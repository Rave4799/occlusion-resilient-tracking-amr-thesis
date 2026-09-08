"""
depth.py — Depth Anything V2 relative depth inference.

STATUS: VALIDATED. Ported from depth_track_annotation.py, with a Day 2
fix (see note below) for a transformers library API change.

Role: produce a raw relative depth map per frame. This module does NOT
classify spatial bins and does NOT touch tracker output — see
association.py for the per-track aggregation + near/mid/far binning
that consumes this module's output.

Depth is RELATIVE, not metric (DepthAnythingV2-S vs KITTI: AbsRel=0.056
— justified for ordering/zone classification only). ISO 3691-4 metric
thresholds do not apply; documented as a known limitation, not a gap
to close (thesis §5.5 / §6.6).

DAY 2 FIX — result["depth"] vs result["predicted_depth"]:
  The original code read result["depth"] as the raw depth array. In
  the transformers version installed for this repo, the HF
  depth-estimation pipeline's "depth" key now returns an already
  0-255 normalized, 8-bit PIL Image (the visualization-ready output),
  NOT the raw relative depth values. The true raw output is under a
  separate key, "predicted_depth", as a torch.float32 tensor.
  Confirmed by direct inspection: result.keys() = ['predicted_depth',
  'depth'], with 'depth' being PIL mode='L' (uint8, 0-255) and
  'predicted_depth' being the raw float32 tensor at native resolution.
  Reading the wrong key produced depth values in the 0-255 range
  instead of the ~0-10ish range the locked P25=1.06/P65=3.61
  thresholds assume — this silently broke every depth_bin assignment
  (everything computed as "far") without crashing, which is why it
  needed direct inspection to catch rather than showing up as an error.
"""
import numpy as np


def load_depth_model(cfg: dict):
    """Load DepthAnythingV2 ViT-Small via the HuggingFace transformers
    pipeline — matches how the original Phase 5 code loaded it (no
    local checkpoint file, downloaded from HF Hub by model name).
    """
    from transformers import pipeline

    # HF pipeline uses the classic convention: 0 = first GPU, -1 = CPU.
    device = 0 if cfg["runtime"]["device"] == "cuda" else -1

    print("[depth] loading DepthAnythingV2 ViT-Small (HF pipeline) ...")
    depth_pipe = pipeline(
        task="depth-estimation",
        model=cfg["depth"]["model_name"],
        device=device,
    )
    print("[depth] model loaded")
    return depth_pipe


def infer_depth_map(depth_pipe, frame_bgr: np.ndarray) -> np.ndarray:
    """Run DepthAnythingV2 on a single frame via the HF pipeline.

    Returns the RAW relative depth map (float32, unnormalized) from
    result["predicted_depth"] — NOT result["depth"], which is a
    normalized 0-255 visualization image in this transformers version
    (see module docstring). Downstream binning (association.py)
    consumes these raw float values.
    """
    import cv2
    from PIL import Image

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # required before inference
    pil_img = Image.fromarray(frame_rgb)

    result = depth_pipe(pil_img)
    raw_depth = result["predicted_depth"].cpu().numpy()  # HxW float32, relative scale

    return raw_depth


def run_depth(frame_bgr: np.ndarray, depth_pipe) -> np.ndarray:
    """Alias matching the run_pipeline.py call signature. Same as
    infer_depth_map — kept as a thin wrapper so the pipeline's stage
    names stay consistent (detect_frame, update_tracks, run_depth, ...).
    """
    return infer_depth_map(depth_pipe, frame_bgr)
