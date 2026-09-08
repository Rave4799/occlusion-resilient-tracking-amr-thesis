"""
Occlusion-Resilient Human Tracking pipeline — source package.

Modules:
  config.py       — loads config.yaml, resolves BASE_DIR-relative paths
  detection.py     — YOLOv10n + Faster R-CNN dual-detector [VALIDATED, DAY 1]
  tracking.py      — DeepSORT + MobileNet Re-ID [VALIDATED, DAY 1]
  depth.py         — Depth Anything V2 per-track annotation [DAY 2]
  association.py   — bbox-direct track-depth binning [DAY 2]
  reasoning.py      — BS-1 trigger, rule template, LLM enrichment [DAY 2]

Status tags above reflect thesis scope: detection and tracking are
validated on a leakage-free held-out set (see thesis Ch.6). depth,
association, and reasoning are specified in the thesis design (Ch.4) as
future work; this repo is where that specification gets implemented and
demonstrated end-to-end, separately from the thesis's own evaluation claims.
"""
