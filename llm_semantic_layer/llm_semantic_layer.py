import pandas as pd, google.generativeai as genai, time, os

# ===== PHASE 7: LLM Semantic Safety Layer =====
# Fires Gemini-2.5-Flash on CRITICAL zone events only (blind-spot reasoning)
# Input: safety_events.csv from Phase 6 Verification Gate
# Output: llm_reasoning_log.csv

API_KEY   = "YOUR_GEMINI_KEY"
MODEL     = "gemini-2.5-flash"
EVENTS_CSV = "/content/drive/MyDrive/verification_gate/safety_events.csv"
OUT_CSV   = "/content/drive/MyDrive/llm_semantic_layer/llm_reasoning_log.csv"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL)

def build_prompt(row):
    return f"""You are a safety reasoning system for an AMR in a warehouse.
Person detected in CRITICAL zone. Frame: {row.frame_idx}, Track: {row.track_id},
Depth score: {row.avg_depth:.2f} (lower=closer). In 2-3 sentences: risk level, recommended AMR action, and why."""

df = pd.read_csv(EVENTS_CSV)
critical = df[df["zone"] == "CRITICAL"].reset_index(drop=True)
log = []
for _, row in critical.iterrows():
    try:
        resp = model.generate_content(build_prompt(row))
        reasoning = resp.text.strip()
    except Exception as e:
        reasoning = f"ERROR: {e}"
    log.append({"frame_idx": row.frame_idx, "track_id": row.track_id,
                 "avg_depth": row.avg_depth, "zone": row.zone, "llm_reasoning": reasoning})
    time.sleep(0.5)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame(log).to_csv(OUT_CSV, index=False)
print(f"Done: {len(log)} CRITICAL events reasoned")
