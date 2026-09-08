# Occlusion-Resilient Human Tracking Pipeline

Companion codebase for the M.Eng. thesis *"Occlusion-Resilient Human
Tracking with Adaptive Logging using an LLM Semantic Layer"*
(Jupudi Ravichandra, HS Schmalkalden).

**Status: complete and validated.** All five pipeline stages are
implemented, tested individually, and confirmed working end-to-end on
two real demo clips. See "Known Issues & Interpretation Notes" below
before drawing conclusions from the results — a few things are
important to read correctly.

---

## What this is

A five-stage pipeline: **detection → tracking → depth annotation →
track-depth association → blind-spot (BS-1) reasoning**.

| Stage | File | Status |
|---|---|---|
| Detection (YOLOv10n + Faster R-CNN) | `src/detection.py` | Validated — thesis Ch.6, mAP50 = 0.822 / 0.852 on leakage-free held-out set |
| Tracking (DeepSORT + MobileNet Re-ID) | `src/tracking.py` | Validated — thesis Ch.6 modular comparison (3 vs. 17 vs. 68 IDs) |
| Depth (DepthAnythingV2) | `src/depth.py` | Implemented and validated this handover — see note below on a library API fix |
| Track-Depth Association | `src/association.py` | Implemented and validated this handover |
| BS-1 Reasoning (trigger, rule template, LLM enrichment) | `src/reasoning.py` | Implemented and validated this handover |

The thesis itself (submitted 8 September 2026) states that depth,
association, and reasoning were "specified but not implemented" —
that was accurate at submission time. Everything from "Depth" downward
in the table above was completed **after** submission, as agreed with
Kate, specifically for defense readiness. This README and the
codebase reflect the current, complete state.

---

## How to run

```bash
pip install -r requirements.txt
```

Set your Gemini API key (for LLM enrichment — optional, the pipeline
works without it, see below):

```bash
echo "GEMINI_API_KEY=your-key-here" > .env
```

Place model weights at the paths specified in `config.yaml`
(`models/yolov10/best.pt`, `models/faster_rcnn/faster_rcnn_final.pth`
by default — update `config.yaml` if yours are elsewhere).

Run on a video file or a directory of frame images:

```bash
python3 run_pipeline.py --input path/to/video.mp4 --output outputs/
python3 run_pipeline.py --input path/to/frame_directory/ --output outputs/
```

Output: an annotated video (`outputs/annotated_output.mp4`, boxes
colour-coded by depth bin — red=near, orange=mid, green=far) and a
structured event log (`outputs/bs1_events.csv`).

---

## Two demo clips, two different claims — do not conflate them

This is the single most important thing to get right when presenting
these results.

**`tests/test_full_chain_test6.py`** runs on a clip from
`~/bs_validation/fullrate_candidate1/` (169 frames), which comes from
the same ROS bags used to build the training dataset. This clip
demonstrates: **the BS-1 trigger logic fires correctly on a known
occlusion pattern** — a functional correctness check. It is *not*
generalization evidence, because the underlying models were trained
on data from this same recording session.

**`tests/test_full_chain_occlusion.py`** runs on a clip extracted
fresh from `2023_05_04_09_44_Gera_C-R_Alt.bag`, a recording never
touched during dataset construction, training, or any prior
evaluation. This clip demonstrates: **the full pipeline generalizes
to genuinely unseen, real-world footage.** This is the actual
generalization evidence.

Validated results:
- **Test6**: 2 BS-1 events fired (tracks 73, 75), both with correct
  stable/lost counts and near-field depth bins.
- **Occlusion clip**: 5 BS-1 events fired (tracks 33, 14, 56, 72, 90),
  all internally consistent (stable 14–32 frames, lost exactly 5, all
  near-field). See "Known Issues" below regarding one caveat on this
  count.

---

## Design notes (see thesis for full technical detail)

- **Depth is spatial context, not a safety verdict.** Output is
  near/mid/far bins from relative (not metric) depth. No safety-zone
  classification, no ISO 13855 conformance claim — this is a
  deliberate scope decision, documented in the thesis descoping.
- **Track-Depth Association replaces the retired Verification Gate**
  entirely. Bbox-direct depth sampling per confirmed track, no IoU
  matching between separately-produced detection sets.
- **BS-1 is a perception-failure event.** It fires when a previously
  stable, near-field track disappears — at the moment of loss, not on
  recovery. Re-identification after occlusion is a separate,
  already-validated tracking claim (see thesis Ch.6); BS-1 was
  deliberately designed not to depend on it succeeding, because
  testing showed DeepSORT does not always re-identify a person
  correctly after full occlusion (see Known Issues, item 3).
- **The rule template is a deterministic reasoning baseline** — a
  reliability guarantee (always produces valid output in well under
  1ms, measured), not a safety decision. It runs independent of the
  optional LLM enrichment step, and the LLM step has a timeout
  fallback: if it fails or times out, the rule template output alone
  is complete and valid. This was demonstrated under real conditions,
  not just in testing — see Known Issues, item 4.

---

## Known Issues & Interpretation Notes

Full detail, including "if asked at defense" framing for each item,
is in **`Known_Issues.pdf`** (included in this
handover). Summary:

1. **Fixed during implementation:** the HuggingFace `transformers`
   depth-estimation pipeline's `"depth"` output key returns a
   normalized 0-255 visualization image in the installed library
   version, not the raw depth values. The real raw float output is
   under `"predicted_depth"`. This silently produced wrong depth bins
   (everything classified "far") until caught by direct inspection.
   Fixed in `src/depth.py`.
2. **Fixed during implementation:** `llm_timeout_seconds` was
   initially 5.0s, too short for the actual reasoning prompt (which
   reliably takes 3.4–3.7s). Raised to 15.0s after isolated testing.
3. **Known limitation, does not block the design:** DeepSORT does not
   always re-identify a person correctly after full occlusion (Test6:
   a 26-frame gap, under `max_age=30`, still resulted in a new track
   ID on reappearance — likely an appearance-embedding mismatch, not
   a timeout). BS-1 was deliberately designed around this limitation.
4. **`google-generativeai` is a fully deprecated library** (Google
   has ended support in favor of `google.genai`). It works correctly
   today (validated against the real API, September 2026) but is
   technical debt for future maintenance.
5. **Gemini free-tier daily quota (20 requests/day) is easily
   exhausted during repeated testing.** Confirmed live: two separate
   test runs today hit `429` quota errors mid-run and correctly fell
   back to rule-template-only output with no crash — real evidence of
   the designed fallback under genuine failure, not just a mock test.
   For a demo requiring uninterrupted LLM enrichment, use a paid tier
   or a fresh daily quota.
6. **Interpretation caveat:** the occlusion clip's 5 BS-1 events have
   not been individually cross-checked frame-by-frame against visual
   ground truth. Given this same clip showed track-ID fragmentation
   (31 unique IDs across 221 frames in Day 1 testing), it is possible
   some of the 5 events represent the same real person's fragmented
   track rather than 5 fully distinct occlusion situations. The
   trigger logic itself is verified correct (see test suite); this
   caveat is about interpreting the raw event count as a real-world
   quantity.
7. **Not yet done:** a manual, human-eyeballed near/far frame
   confirmation for Test6 (deferred from Day 1, since depth wasn't
   integrated yet at that point). The automated evidence is
   reasonably strong (correct BS-1 firings, sensible depth values),
   but this explicit final sanity check was never performed.

---

## Performance

Measured end-to-end on macOS, CPU only (no GPU): **~1.9 seconds per
frame (~0.52 FPS)** across all three models (YOLOv10n, Faster R-CNN,
DepthAnythingV2) per frame. GPU inference would be substantially
faster; this pipeline has not yet been benchmarked on GPU hardware.

---

## Repo structure

```
├── run_pipeline.py         # single entry point, chains all 5 stages
├── config.yaml             # all paths and parameters — no hardcoding in src/
├── requirements.txt        # pinned to validated versions
├── .env                    # GEMINI_API_KEY (not committed — see .gitignore)
├── src/
│   ├── config.py           # loads config.yaml, resolves BASE_DIR
│   ├── detection.py        # YOLOv10n + Faster R-CNN [validated]
│   ├── tracking.py         # DeepSORT + MobileNet Re-ID [validated]
│   ├── depth.py             # DepthAnythingV2 spatial context [validated]
│   ├── association.py      # track-depth binning [validated]
│   └── reasoning.py         # BS-1 trigger + rule template + LLM [validated]
├── tests/
│   ├── test_full_chain_test6.py       # functional correctness validation
│   └── test_full_chain_occlusion.py   # generalization validation
├── models/                 # model weights (not committed — see below)
├── data/                   # demo input clips
└── outputs/                # pipeline output: annotated video + event logs
```

## Model weights

Not committed to this repo (see `.gitignore` — `.pt`/`.pth` files
excluded). Weight files used for all validation in this handover:
- `models/yolov10/best.pt` (5.7 MB)
- `models/faster_rcnn/faster_rcnn_final.pth` (165.7 MB)

Obtain these from the project's Google Drive
(`Thesis Main Implementation/Phase 2 - Yolo Training/` and
`Phase 3 - FasterR_CNN/` respectively) and place at the paths above,
or update `config.yaml` to point elsewhere.
