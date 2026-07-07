import os, glob, cv2, re, csv, time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

# ---- Config ----
BEST_PT     = 'yolo_training/best.pt'
FRAMES_DIR  = 'frames/'           # sorted jpg frames
OUT_DIR     = 'results/deepsort/'
CONF        = 0.4
MAX_AGE     = 30    # Kalman filter: keep lost track alive N frames
N_INIT      = 3     # detections to confirm a track
MAX_COSINE  = 0.4   # Re-ID similarity threshold
NN_BUDGET   = 100   # max embedding history per track

os.makedirs(OUT_DIR, exist_ok=True)

model   = YOLO(BEST_PT)
tracker = DeepSort(
    max_age=MAX_AGE, n_init=N_INIT,
    nms_max_overlap=1.0,
    max_cosine_distance=MAX_COSINE,
    nn_budget=NN_BUDGET,
    embedder='mobilenet', half=True, embedder_gpu=True
)

def get_color(tid):
    seed = abs(hash(str(tid))) % (2**31)
    rng  = np.random.RandomState(seed)
    return tuple(int(x) for x in rng.randint(50, 255, 3))

frames = sorted(glob.glob(FRAMES_DIR + '*.jpg'))
track_log, unique_ids = [], set()

for i, img_path in enumerate(tqdm(frames)):
    frame = cv2.imread(img_path)
    if frame is None: continue
    results = model(frame, conf=CONF, verbose=False)[0]
    dets = []
    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        dets.append(([x1,y1,x2-x1,y2-y1], float(box.conf[0]), 'person'))
    tracks = tracker.update_tracks(dets, frame=frame)
    for t in tracks:
        if not t.is_confirmed(): continue
        tid = str(t.track_id)
        unique_ids.add(tid)
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1,y1), (x2,y2), get_color(tid), 2)
        cv2.putText(frame, f'ID:{tid}', (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_color(tid), 2)
        track_log.append((i, tid, x1,y1,x2,y2))
    cv2.imwrite(f'{OUT_DIR}/frame_{i:04d}.jpg', frame)

with open(f'{OUT_DIR}/track_log.csv','w',newline='') as f:
    csv.writer(f).writerows([('frame','id','x1','y1','x2','y2')] + track_log)
print(f'Done. Unique IDs: {len(unique_ids)}, Records: {len(track_log)}')
