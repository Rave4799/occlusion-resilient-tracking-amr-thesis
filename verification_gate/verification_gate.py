import pandas as pd
import numpy as np
import cv2, os, glob
from tqdm import tqdm

# ===== PHASE 6: Verification Gate =====
# Fuses Deep SORT track_log + DepthAnythingV2 depth_log
# Classifies each tracked person into ISO 10218 safety zones
# Outputs: safety_events.csv + annotated frames + demo video

TRACK_CSV  = '/content/drive/MyDrive/deepsort_output/track_log.csv'
DEPTH_CSV  = '/content/drive/MyDrive/depth_output/depth_log.csv'
FRAMES_DIR = '/content/drive/MyDrive/depth_output/annotated_frames'
OUT_DIR    = '/content/drive/MyDrive/verification_gate'

ZONE_COLORS = {'CRITICAL': (0,0,255), 'WARNING': (0,165,255), 'SAFE': (0,255,0)}

def compute_iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0

track_df = pd.read_csv(TRACK_CSV)
depth_df  = pd.read_csv(DEPTH_CSV)

merged_rows = []
for frame_idx, t_grp in track_df.groupby("frame_idx"):
    d_grp = depth_df[depth_df.frame_idx == frame_idx]
    for _, t in t_grp.iterrows():
        best_iou, best_d = 0.0, None
        for _, d in d_grp.iterrows():
            iou = compute_iou((t.x1,t.y1,t.x2,t.y2),(d.x1,d.y1,d.x2,d.y2))
            if iou > best_iou: best_iou, best_d = iou, d.avg_depth
        merged_rows.append({"frame_idx":int(frame_idx),"track_id":str(t.track_id),
            "x1":int(t.x1),"y1":int(t.y1),"x2":int(t.x2),"y2":int(t.y2),
            "iou_match":round(best_iou,3),"avg_depth":round(float(best_d),4) if best_d else None})

df = pd.DataFrame(merged_rows).dropna(subset=["avg_depth"])
p25, p65 = np.percentile(df.avg_depth, 25), np.percentile(df.avg_depth, 65)
df["zone"] = df.avg_depth.apply(lambda d: "CRITICAL" if d<p25 else ("WARNING" if d<p65 else "SAFE"))

os.makedirs(f"{OUT_DIR}/annotated_frames", exist_ok=True)
for fidx, grp in tqdm(df.groupby("frame_idx")):
    src = f"{FRAMES_DIR}/frame_{int(fidx):04d}.jpg"
    if not os.path.exists(src): continue
    frame = cv2.imread(src)
    for _, r in grp.iterrows():
        c = ZONE_COLORS[r.zone]
        cv2.rectangle(frame,(int(r.x1),int(r.y1)),(int(r.x2),int(r.y2)),c,3)
        lbl = f"ID:{r.track_id} {r.zone} d:{r.avg_depth:.1f}"
        cv2.putText(frame,lbl,(int(r.x1)+2,int(r.y1)-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
    cv2.imwrite(f"{OUT_DIR}/annotated_frames/frame_{int(fidx):04d}.jpg", frame)

df[df.zone!="SAFE"].to_csv(f"{OUT_DIR}/safety_events.csv", index=False)
print(f"Done: {len(df)} detections, CRITICAL={len(df[df.zone=='CRITICAL'])}, WARNING={len(df[df.zone=='WARNING'])}")
