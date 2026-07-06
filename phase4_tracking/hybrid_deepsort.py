"""
hybrid_deepsort.py - Hybrid Detector + DeepSORT Tracker
Phase 4 Final | Thesis: Occlusion-Resilient Human Tracking
Jupudi Ravichandra (315502) | HS Schmalkalden

PIPELINE:
  1. YOLO runs on every frame (fast primary detector)
  2. If any det conf < TRIGGER_CONF -> R-CNN also runs as verifier
  3. IoU-match boxes -> AGREED / VERIFIED / YOLO_ONLY / RCNN_ONLY
  4. Only AGREED + VERIFIED + RCNN_ONLY fed to DeepSORT (YOLO_ONLY discarded)
  5. DeepSORT Config C: MAX_COSINE=0.25, NN_BUDGET=150, MAX_AGE=30, N_INIT=3
"""

import os, glob, re, cv2, csv, time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

YOLO_CONF    = 0.40
TRIGGER_CONF = 0.70
RCNN_SCORE   = 0.50
IOU_MATCH    = 0.30
MAX_AGE      = 30
N_INIT       = 3
MAX_COSINE   = 0.25
NN_BUDGET    = 150
NMS_OVERLAP  = 0.7
MAX_FRAMES   = 300
MIN_TRACK_LEN = 5
FPS_OUT      = 10
FRAME_W      = 640
FRAME_H      = 480

YOLO_WEIGHTS = "/content/drive/MyDrive/thesis /yolo_runs/yolov10_custom/weights/best.pt"
RCNN_WEIGHTS = "/content/drive/MyDrive/thesis /faster_rcnn_best.pth"
FRAMES_DIR   = "/content/drive/MyDrive/thesis /depth_output/annotated_frames"
FALLBACK_DIR = "/content/dataset_raw/labeled_images"
OUT_DIR      = "/content/drive/MyDrive/thesis /testing_results/hybrid_deepsort"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR + "/frames", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: " + device)

print("Loading YOLO...")
yolo = YOLO(YOLO_WEIGHTS)

print("Loading Faster R-CNN...")
def load_rcnn(path, dev):
    m = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.to(dev); m.eval()
    return m

rcnn = load_rcnn(RCNN_WEIGHTS, device)

print("Loading DeepSORT (Config C)...")
tracker = DeepSort(
    max_age=MAX_AGE, n_init=N_INIT, nms_max_overlap=NMS_OVERLAP,
    max_cosine_distance=MAX_COSINE, nn_budget=NN_BUDGET,
    embedder="mobilenet", half=True, embedder_gpu=True,
)
print("All models loaded.\n")

def load_frames(primary, fallback, max_n):
    if os.path.isdir(primary):
        imgs = sorted(glob.glob(os.path.join(primary, "*.jpg")))
        if imgs:
            print("Using primary: " + primary)
            return imgs[:max_n]
    print("Fallback: " + fallback)
    imgs = sorted(
        glob.glob(os.path.join(fallback, "*.jpg")),
        key=lambda p: int(re.search(r"_(\d{6})\.jpg", p).group(1))
                      if re.search(r"_(\d{6})\.jpg", p) else 0
    )
    return imgs[:max_n]

frames = load_frames(FRAMES_DIR, FALLBACK_DIR, MAX_FRAMES)
print("Loaded " + str(len(frames)) + " frames")
if not frames:
    raise RuntimeError("No frames found.")

def compute_iou(a, b):
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def get_color(tid):
    seed = abs(hash(str(tid))) % (2**31)
    rng = np.random.RandomState(seed)
    return tuple(int(x) for x in rng.randint(80, 255, 3))

def hybrid_detect(frame):
    res = yolo(frame, conf=YOLO_CONF, verbose=False)[0]
    yolo_dets = []
    for b in res.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
        yolo_dets.append((x1,y1,x2,y2,round(float(b.conf[0]),3)))
    needs_rcnn = any(d[4] < TRIGGER_CONF for d in yolo_dets) or len(yolo_dets) == 0
    if needs_rcnn:
        t = TF.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
        with torch.no_grad():
            pred = rcnn([t])[0]
        rcnn_dets = []
        for box,score,label in zip(pred["boxes"],pred["scores"],pred["labels"]):
            if label.item()==1 and score.item()>=RCNN_SCORE:
                x1,y1,x2,y2 = map(int,box.tolist())
                rcnn_dets.append((x1,y1,x2,y2,round(float(score),3)))
    else:
        rcnn_dets = []
    matched_rcnn = set()
    confirmed, discarded = [], []
    for yd in yolo_dets:
        best_iou, best_ri = 0.0, -1
        for ri, rd in enumerate(rcnn_dets):
            iou = compute_iou(yd[:4], rd[:4])
            if iou > best_iou: best_iou, best_ri = iou, ri
        if best_iou >= IOU_MATCH:
            matched_rcnn.add(best_ri)
            confirmed.append(yd)
        else:
            discarded.append(yd)
    for ri, rd in enumerate(rcnn_dets):
        if ri not in matched_rcnn:
            confirmed.append(rd)
    return confirmed, discarded, len(rcnn_dets) > 0

track_log, unique_ids, track_lens, ids_per_frm = [], set(), {}, []
discarded_total = rcnn_trigger_count = 0
video_out = None
t0 = time.time()

for i, img_path in enumerate(tqdm(frames, desc="Hybrid+DeepSORT", ncols=70)):
    frame = cv2.imread(img_path)
    if frame is None: continue
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    confirmed, discarded, rcnn_used = hybrid_detect(frame)
    discarded_total += len(discarded)
    if rcnn_used: rcnn_trigger_count += 1
    dets = [([x1,y1,x2-x1,y2-y1], cf, "person") for (x1,y1,x2,y2,cf) in confirmed]
    tracks = tracker.update_tracks(dets, frame=frame)
    frame_ids = set()
    for t in tracks:
        if not t.is_confirmed(): continue
        tid = int(t.track_id)
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        unique_ids.add(tid); frame_ids.add(tid)
        track_lens[tid] = track_lens.get(tid, 0) + 1
        track_log.append((i, tid, x1, y1, x2, y2))
        col = get_color(tid)
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
        lbl = "ID:" + str(tid)
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1,y1-th-8), (x1+tw+4,y1), col, -1)
        cv2.putText(frame, lbl, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    for (x1,y1,x2,y2,cf) in discarded:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (100,100,100), 1)
        cv2.putText(frame, "FP", (x1,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)
    ids_per_frm.append(len(frame_ids))
    rcnn_flag = "R:ON" if rcnn_used else "R:OFF"
    info_text = "Fr:" + str(i) + " IDs:" + str(len(frame_ids)) + " " + rcnn_flag
    cv2.putText(frame, info_text, (6,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, "Hybrid+DeepSORT | Config C", (6,40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,255,180), 1)
    cv2.imwrite(OUT_DIR + "/frames/frame_" + str(i).zfill(4) + ".jpg", frame)
    if video_out is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_out = cv2.VideoWriter(OUT_DIR + "/tracked_output.mp4", fourcc, FPS_OUT, (FRAME_W, FRAME_H))
    video_out.write(frame)

if video_out: video_out.release()
elapsed = time.time() - t0

with open(OUT_DIR + "/track_log.csv", "w", newline="") as f:
    csv.writer(f).writerows([("frame","id","x1","y1","x2","y2")] + track_log)

total_ids   = len(unique_ids)
short_tracks = sum(1 for v in track_lens.values() if v < MIN_TRACK_LEN)
avg_len     = sum(track_lens.values())/len(track_lens) if track_lens else 0
avg_ids_frm = sum(ids_per_frm)/len(ids_per_frm) if ids_per_frm else 0

print("\n" + "="*60)
print("  HYBRID + DEEPSORT RESULTS")
print("="*60)
print("  Frames     : " + str(len(frames)) + " @ " + str(round(len(frames)/elapsed,1)) + " fps")
print("  R-CNN used : " + str(rcnn_trigger_count) + " frames")
print("  FP blocked : " + str(discarded_total) + " detections discarded")
print("  Unique IDs : " + str(total_ids))
print("  Ghost trks : " + str(short_tracks))
print("  Avg len    : " + str(round(avg_len,1)) + " frames")
print("  Avg IDs/fr : " + str(round(avg_ids_frm,2)))
print("  Video      : " + OUT_DIR + "/tracked_output.mp4")
print("="*60)
