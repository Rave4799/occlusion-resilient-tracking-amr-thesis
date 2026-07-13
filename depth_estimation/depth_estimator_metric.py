# depth_estimator_metric.py - Phase 5b: DepthAnythingV2 + Metric Calibration
# Converts relative depth to metric (metres) using RealSense scale+shift calibration
# Applies 3D back-projection and ISO 10218 / ISO/TS 15066 safety zones

import os, glob, cv2, re, csv, json
import numpy as np
from transformers import pipeline
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
IMG_DIR      = '/content/dataset_raw/labeled_images'
YOLO_WEIGHTS = '/content/drive/MyDrive/yolo_runs/yolov10_custom/weights/best.pt'
CALIB_JSON   = '/content/drive/MyDrive/depth_calibration.json'
OUTPUT_DIR   = '/content/drive/MyDrive/depth_output_metric'
CONF         = 0.4
NUM_FRAMES   = 300

# ── Camera intrinsics (RealSense D435, confirmed from bag file) ────────────────
FX, FY = 604.867, 604.867   # focal lengths in pixels
CX, CY = 320.66,  240.0     # principal point in pixels

# ── Safety zone thresholds — ISO 10218 / ISO/TS 15066 ─────────────────────────
ZONE_COLORS = {
    'CRITICAL': (0,   0, 255),   # red    — Z < 0.5 m  → immediate stop
    'WARNING':  (0, 165, 255),   # orange — 0.5–1.5 m  → slow down
    'SAFE':     (0, 255,   0),   # green  — Z > 1.5 m  → normal operation
}

def classify_zone(z_m):
    if   z_m < 0.5:  return 'CRITICAL'
    elif z_m < 1.5:  return 'WARNING'
    else:             return 'SAFE'

# ── Load calibration params (s, t) from Drive ─────────────────────────────────
with open(CALIB_JSON) as f:
    calib = json.load(f)
SCALE = calib['scale']
SHIFT = calib['shift']
print(f"Calibration: s={SCALE:.6f}  t={SHIFT:.6f} m  |  RMSE={calib['rmse_cm']:.2f} cm")

os.makedirs(OUTPUT_DIR + '/depth_maps',       exist_ok=True)
os.makedirs(OUTPUT_DIR + '/annotated_frames', exist_ok=True)

# ── Load models ───────────────────────────────────────────────────────────────
depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0
)
det_model = YOLO(YOLO_WEIGHTS)

frames = sorted(
    glob.glob(IMG_DIR + '/*.jpg'),
    key=lambda p: int(re.search(r'_(\d{6})\.jpg', p).group(1))
                  if re.search(r'_(\d{6})\.jpg', p) else 0
)[:NUM_FRAMES]

depth_log = []

for i, fp in enumerate(tqdm(frames, desc="Processing")):
    fr = cv2.imread(fp)

    # ── Step 1: Relative depth from DepthAnythingV2 ───────────────────────────
    pil_rgb = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    dn_rel  = np.array(depth_pipe(pil_rgb)['depth']).astype(np.float32)

    # ── Step 2: Apply metric calibration  Z_m = s * Z_rel + t ────────────────
    dn_met = np.clip(SCALE * dn_rel + SHIFT, 0.0, 10.0)   # metres, clipped to 10 m

    # ── Step 3: Save inferno depth map (relative, for visualisation) ──────────
    n = cv2.normalize(dn_rel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(
        f'{OUTPUT_DIR}/depth_maps/depth_{i:04d}.jpg',
        cv2.applyColorMap(n, cv2.COLORMAP_INFERNO)
    )

    # ── Step 4: Per-detection — metric depth, 3D position, safety zone ────────
    for box in det_model(fr, conf=CONF, verbose=False)[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_val = round(float(box.conf[0]), 3)

        # Median depth at bbox centre region (robust vs. noisy edges)
        cx_b = (x1 + x2) // 2
        cy_b = (y1 + y2) // 2
        mh   = max(1, (y2 - y1) // 4)
        mw   = max(1, (x2 - x1) // 4)
        crop = dn_met[max(0, cy_b-mh):cy_b+mh, max(0, cx_b-mw):cx_b+mw]
        Z    = float(np.median(crop)) if crop.size > 0 else 0.0

        # 3D back-projection into camera frame
        X = (cx_b - CX) * Z / FX
        Y = (cy_b - CY) * Z / FY

        zone  = classify_zone(Z)
        color = ZONE_COLORS[zone]

        depth_log.append((
            i, x1, y1, x2, y2, conf_val,
            round(Z, 3), round(X, 3), round(Y, 3), zone
        ))

        # ── Step 5: Annotate frame with zone + metric values ──────────────────
        cv2.rectangle(fr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(fr, f'{zone}  {Z:.2f}m',
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(fr, f'X:{X:.2f}m  Y:{Y:.2f}m',
                    (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    cv2.imwrite(f'{OUTPUT_DIR}/annotated_frames/frame_{i:04d}.jpg', fr)

# ── Save metric CSV log ───────────────────────────────────────────────────────
header = ['frame_idx', 'x1', 'y1', 'x2', 'y2', 'conf',
          'Z_m', 'X_m', 'Y_m', 'zone']
with open(OUTPUT_DIR + '/depth_log_metric.csv', 'w', newline='') as f:
    csv.writer(f).writerows([header] + depth_log)

print(f"\nDone. {len(depth_log)} detections logged.")
print(f"Zone breakdown:")
for z in ['CRITICAL', 'WARNING', 'SAFE']:
    count = sum(1 for row in depth_log if row[-1] == z)
    print(f"  {z}: {count} ({100*count/max(len(depth_log),1):.1f}%)")
print(f"Output: {OUTPUT_DIR}")
