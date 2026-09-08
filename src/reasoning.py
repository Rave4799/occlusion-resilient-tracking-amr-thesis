"""
reasoning.py — BS-1 blind-spot reasoning: trigger, rule template, LLM enrichment.

STATUS: ALL FOUR reasoning.py functions complete (track_history — Day 1,
  bs1_trigger/rule_template/llm_enrichment — Day 2). This module is done.

BS-1 TRIGGER DEFINITION (locked, thesis §4.6.1 / Ch.7 reckoning):
  Fires when a track was stable for >= n_stable_frames consecutive frames,
  then absent for >= n_lost_frames consecutive frames, AND its last
  observed depth_bin was "near".

  This is a PERCEPTION-FAILURE event: it fires AT the moment of loss,
  not on recovery. Re-identification after occlusion is a SEPARATE,
  already-validated claim (tracking.py / thesis Ch.6 modular comparison)
  — do not couple BS-1 firing to whether the track is later recovered.

  REAL-DATA CHECK (Day 1, Test6): the actual occlusion behind the
  clothing rack produced a 26-frame zero-detection gap (frames 90-115,
  ~1.7s at 15fps) — just under DeepSORT's max_age=30, yet the person
  still received a new track ID on reappearance. This confirms BS-1
  must NOT depend on successful re-identification. n_lost_frames=5
  (config.yaml) is comfortably conservative against this real 26-frame
  gap — it fires well before recovery is even possible, which is correct.

DAY 2 FIX — frames_stable_before_loss:
  update_track_history() resets consecutive_present_frames to 0 the
  instant a track goes absent (correct — a track must re-stabilize from
  scratch after a loss, per §4.6.1). But that means the "was it stable
  for >= n_stable_frames" question can't be answered from
  consecutive_present_frames alone once the track is already absent —
  the count is gone by then. frames_stable_before_loss captures that
  count at the exact present->absent transition frame, so bs1_trigger()
  has something to check against during the loss itself.

TERMINOLOGY: the rule_template is a "deterministic reasoning baseline" /
"functional baseline" — NEVER "safety floor" (safety-decision framing is
out of scope per the thesis descope). It is a reliability guarantee
(always produces valid output), not a safety verdict.

DAY 2 REMAINING TASKS: none — reasoning.py is complete. Remaining Day 2
  work is testing bs1_trigger against real clips and wiring run_pipeline.py.
"""
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Track history — DONE (Day 1, extended Day 2)
# ---------------------------------------------------------------------------
@dataclass
class TrackHistoryEntry:
    """Per-track state needed to evaluate the BS-1 trigger condition.

    One instance per track_id, held in the track_history dict that
    persists across frames (created once per clip/stream, passed into
    update_track_history() every frame).
    """
    track_id: int
    consecutive_present_frames: int = 0
    consecutive_absent_frames: int = 0
    last_depth_bin: str = None       # "near" | "mid" | "far" | None
    last_avg_depth: float = None
    last_seen_frame: int = None      # frame index this track was last confirmed present
    frames_stable_before_loss: int = None  # captured at present->absent transition (Day 2)
    bs1_fired: bool = False          # prevents re-firing every frame while still lost


def update_track_history(track_history: dict, tracks_with_depth: list, frame_idx: int) -> dict:
    """Update per-track presence/absence counters for this frame.

    track_history: dict[track_id -> TrackHistoryEntry], persists across
        frames. Pass an empty {} on the first frame of a clip/stream.
    tracks_with_depth: list[association.TrackWithDepth] — this frame's
        confirmed tracks, already enriched with depth_bin. An empty list
        means no one was detected this frame at all.
    frame_idx: current frame index, stored as last_seen_frame on presence.

    Returns the same track_history dict, mutated in place.

    Behavior:
      - A track_id present this frame: increment consecutive_present_frames,
        reset consecutive_absent_frames to 0, update last_depth_bin /
        last_avg_depth / last_seen_frame. If it was previously absent
        (a fresh reappearance), reset bs1_fired to False so a future
        loss of this track can trigger BS-1 again.
      - A track_id NOT present this frame (existed before, missing now):
        on the FIRST frame of this absence, capture
        frames_stable_before_loss = the consecutive_present_frames value
        it had just before this reset. Then increment
        consecutive_absent_frames and reset consecutive_present_frames
        to 0.
      - A track_id seen for the first time ever: create a new entry.
    """
    present_ids = {t.track_id for t in tracks_with_depth}

    for t in tracks_with_depth:
        is_new = t.track_id not in track_history
        if is_new:
            track_history[t.track_id] = TrackHistoryEntry(track_id=t.track_id)

        entry = track_history[t.track_id]

        # Reappearance after a loss: allow BS-1 to fire again on a future loss.
        if not is_new and entry.consecutive_absent_frames > 0:
            entry.bs1_fired = False

        entry.consecutive_present_frames += 1
        entry.consecutive_absent_frames = 0
        entry.last_depth_bin = t.depth_bin
        entry.last_avg_depth = t.avg_depth
        entry.last_seen_frame = frame_idx

    for track_id, entry in track_history.items():
        if track_id not in present_ids:
            if entry.consecutive_absent_frames == 0:
                # First frame of this absence — capture the stable count
                # before it gets reset below.
                entry.frames_stable_before_loss = entry.consecutive_present_frames

            entry.consecutive_absent_frames += 1
            entry.consecutive_present_frames = 0

    return track_history


# ---------------------------------------------------------------------------
# BS-1 event — DONE (Day 2)
# ---------------------------------------------------------------------------
@dataclass
class BS1Event:
    """A single BS-1 event, matching the fixed rule_template structure."""
    event_type: str = "BS-1"
    track_id: int = None
    trigger_frame: int = None
    last_seen_frame: int = None
    frames_stable_before_loss: int = None
    frames_lost: int = None
    last_depth_bin: str = None
    last_avg_depth: float = None
    reasoning: str = (
        "Track lost in near-field after stable observation — "
        "possible occlusion behind static obstacle."
    )
    llm_enrichment: str = None   # filled in by llm_enrichment(), optional


def bs1_trigger(track_history: dict, tracks_with_depth: list, frame_idx: int, cfg: dict):
    """Check whether any track in track_history now satisfies the BS-1
    condition (stable >= n_stable_frames, then lost >= n_lost_frames,
    last depth_bin == 'near'). Returns a BS1Event or None.

    Call update_track_history() BEFORE this each frame — this function
    reads track_history state, it does not update presence/absence
    counters itself (it does set bs1_fired=True on the entry that fires,
    which is the one piece of state this function owns).

    tracks_with_depth is accepted for interface symmetry with
    update_track_history but is not used directly here — all decisions
    are made from track_history, which already reflects this frame's
    update. Kept as a parameter in case future BS event types (BS-2,
    BS-3 — thesis §4.6, not implemented in this repo) need direct
    access to current-frame tracks alongside history.

    Only ONE event is returned per call, even if multiple tracks
    qualify in the same frame (returns the first found, in track_history
    iteration order). This matches run_pipeline.py's current single-event
    call site. Extending to multiple simultaneous events is future work
    if a scenario requires it.
    """
    n_stable = cfg["reasoning"]["n_stable_frames"]
    n_lost = cfg["reasoning"]["n_lost_frames"]

    for track_id, entry in track_history.items():
        if entry.bs1_fired:
            continue  # already fired for this loss episode

        if entry.consecutive_absent_frames < n_lost:
            continue  # not lost long enough yet

        if entry.frames_stable_before_loss is None or entry.frames_stable_before_loss < n_stable:
            continue  # wasn't stable long enough before it disappeared

        if entry.last_depth_bin != "near":
            continue  # only near-field losses are perception-failure-relevant

        # All three conditions met — fire.
        entry.bs1_fired = True

        return BS1Event(
            track_id=track_id,
            trigger_frame=frame_idx,
            last_seen_frame=entry.last_seen_frame,
            frames_stable_before_loss=entry.frames_stable_before_loss,
            frames_lost=entry.consecutive_absent_frames,
            last_depth_bin=entry.last_depth_bin,
            last_avg_depth=entry.last_avg_depth,
        )

    return None


# ---------------------------------------------------------------------------
# Rule template — DONE (Day 2)
# ---------------------------------------------------------------------------
def rule_template(event: "BS1Event") -> dict:
    """Convert a BS1Event into the fixed, deterministic log entry
    structure. This must run in < 0.001s and never depend on external
    calls — it is the reliability baseline.

    Deterministic by construction: every field is copied directly from
    the event (which was itself built entirely from track_history state
    in bs1_trigger — no randomness, no I/O, no external calls anywhere
    in this path). The 'reasoning' string is the fixed template text
    already set as BS1Event's default — this function does not generate
    or vary it.

    llm_enrichment is initialized to None here; llm_enrichment() (next
    Day 2 task) fills it in afterward as a separate, optional step. If
    that step is skipped or fails, this dict alone is a complete,
    valid log entry.
    """
    return {
        "event_type": event.event_type,
        "track_id": event.track_id,
        "trigger_frame": event.trigger_frame,
        "last_seen_frame": event.last_seen_frame,
        "frames_stable_before_loss": event.frames_stable_before_loss,
        "frames_lost": event.frames_lost,
        "last_depth_bin": event.last_depth_bin,
        "last_avg_depth": event.last_avg_depth,
        "reasoning": event.reasoning,
        "llm_enrichment": event.llm_enrichment,  # None unless llm_enrichment() has run
    }


# ---------------------------------------------------------------------------
# LLM enrichment — DONE (Day 2)
# ---------------------------------------------------------------------------
def llm_enrichment(entry: dict, cfg: dict) -> dict:
    """Call the configured LLM (Gemini) once to append a natural-language
    explanation to the entry. On timeout/failure, return entry unchanged —
    the rule_template output alone is already complete and valid.

    API key: read from the GEMINI_API_KEY environment variable. This
    function never hardcodes or logs the key itself.

    Returns a NEW dict (does not mutate the input entry) with
    'llm_enrichment' set to the model's response text on success, or
    left as whatever it was on the input entry (typically None) if the
    LLM is disabled, the call fails, or it times out. This function
    never raises — every failure path degrades to returning the entry
    unchanged, matching the rule_template's role as the safety-net
    reliability baseline.
    """
    if not cfg["reasoning"]["llm_enabled"]:
        return entry

    result = dict(entry)  # never mutate the caller's dict

    try:
        import os
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("[reasoning] GEMINI_API_KEY not set — skipping LLM enrichment, "
                  "rule_template output stands alone.")
            return result

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(cfg["reasoning"]["llm_model"])

        prompt = (
            f"A perception system on a mobile robot lost track of a person "
            f"who was previously stable for {entry['frames_stable_before_loss']} "
            f"frames and has now been absent for {entry['frames_lost']} frames. "
            f"Their last known position was in the '{entry['last_depth_bin']}' "
            f"spatial zone. In one or two plain sentences, describe this as a "
            f"perception event — do not speculate about the person's safety or "
            f"intentions, only describe what the sensor observed."
        )

        timeout_s = cfg["reasoning"]["llm_timeout_seconds"]
        response = model.generate_content(
            prompt,
            request_options={"timeout": timeout_s},
        )

        result["llm_enrichment"] = response.text.strip()

    except Exception as e:
        # Any failure (timeout, network, API error, missing package) —
        # log it and degrade gracefully. rule_template's output already
        # in `result` remains complete and valid without this field.
        print(f"[reasoning] LLM enrichment failed or timed out ({e}); "
              f"continuing with rule_template output only.")

    return result
