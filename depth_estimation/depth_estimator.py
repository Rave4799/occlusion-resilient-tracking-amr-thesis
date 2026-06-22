# depth_estimator.py - Phase 5: DepthAnythingV2-Small
import os, glob, cv2, re, csv
import numpy as np
from transformers import pipeline
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm

IMG_DIR = '/content/dataset_raw/labeled_images'
YOLO_WEIGHTS = '/content/drive/MyDrive/yolo_runs/yolov10_custom/weights/best.pt'
OUTPUT_DIR = '/content/drive/MyDrive/depth_output'
CONF = 0.4
NUM_FRAMES = 300

os.makedirs(OUTPUT_DIR+'/depth_maps', exist_ok=True)
os.makedirs(OUTPUT_DIR+'/annotated_frames', exist_ok=True)

depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
model = YOLO(YOLO_WEIGHTS)

frames = sorted(glob.glob(IMG_DIR+'/*.jpg'),
    key=lambda p: int(re.search(r'_(\d{6})\.jpg',p).group(1)) if re.search(r'_(\d{6})\.jpg',p) else 0)[:NUM_FRAMES]

depth_log = []
for i,fp in enumerate(tqdm(frames)):
    fr = cv2.imread(fp)
    dn = np.array(depth_pipe(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))['depth'])
    n = cv2.normalize(dn,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f'{OUTPUT_DIR}/depth_maps/depth_{i:04d}.jpg', cv2.applyColorMap(n,cv2.COLORMAP_INFERNO))
    H,W = dn.shape
    for box in model(fr,conf=CONF,verbose=False)[0].boxes:
        x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
        crop=dn[max(0,y1):min(H,y2),max(0,x1):min(W,x2)]
        avg_d=float(np.mean(crop)) if crop.size>0 else 0.0
        depth_log.append((i,x1,y1,x2,y2,round(float(box.conf[0]),3),round(avg_d,4)))
        cv2.rectangle(fr,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(fr,f'd:{avg_d:.2f}',(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
    cv2.imwrite(f'{OUTPUT_DIR}/annotated_frames/frame_{i:04d}.jpg',fr)

with open(OUTPUT_DIR+'/depth_log.csv','w',newline='') as f:
    csv.writer(f).writerows([['frame_idx','x1','y1','x2','y2','conf','avg_depth']]+depth_log)
print(f"Done. {len(depth_log)} detections logged.")
