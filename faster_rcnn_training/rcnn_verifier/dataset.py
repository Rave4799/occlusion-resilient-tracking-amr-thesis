"""Phase 3 - Faster R-CNN Dataset
"""
import os, torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class HumanDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms=None):
        self.img_dir=img_dir; self.lbl_dir=lbl_dir; self.transforms=transforms
        self.imgs=sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img_name=self.imgs[idx]
        img=Image.open(os.path.join(self.img_dir,img_name)).convert("RGB")
        w,h=img.size
        lbl_path=os.path.join(self.lbl_dir,os.path.splitext(img_name)[0]+".txt")
        boxes,labels=[],[]
        if os.path.exists(lbl_path):
            for line in open(lbl_path).readlines():
                p=line.strip().split()
                if len(p)<5: continue
                xc,yc,bw,bh=float(p[1]),float(p[2]),float(p[3]),float(p[4])
                x1=max(0.0,(xc-bw/2)*w); y1=max(0.0,(yc-bh/2)*h)
                x2=min(float(w),(xc+bw/2)*w); y2=min(float(h),(yc+bh/2)*h)
                if x2>x1 and y2>y1:
                    boxes.append([x1,y1,x2,y2]); labels.append(1)
        boxes=torch.tensor(boxes,dtype=torch.float32) if boxes else torch.zeros((0,4),dtype=torch.float32)
        labels=torch.tensor(labels,dtype=torch.int64) if labels else torch.zeros((0,),dtype=torch.int64)
        target={"boxes":boxes,"labels":labels,"image_id":torch.tensor([idx])}
        if self.transforms: img=self.transforms(img)
        return img,target

def get_transform():
    return transforms.Compose([transforms.ToTensor()])
def collate_fn(batch):
    return tuple(zip(*batch))
