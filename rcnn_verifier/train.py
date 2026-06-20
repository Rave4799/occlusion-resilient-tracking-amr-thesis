"""Phase 3 - Faster R-CNN Training
Loss: 0.1216->0.0508, 10 epochs, T4, 5773 images
"""
import os, time, torch
from torch.utils.data import DataLoader, random_split
from dataset import HumanDataset, get_transform, collate_fn
from model import build_model

IMG_DIR="/content/dataset/images"
LBL_DIR="/content/dataset/labels"
SAVE_BEST="/content/drive/MyDrive/faster_rcnn_best.pth"
SAVE_FINAL="/content/drive/MyDrive/faster_rcnn_final.pth"
NUM_EPOCHS=10
BATCH_SIZE=4
LR=0.005

dataset=HumanDataset(IMG_DIR,LBL_DIR,transforms=get_transform())
n=len(dataset); train_ds,val_ds=random_split(dataset,[int(0.8*n),n-int(0.8*n)])
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,collate_fn=collate_fn)
model,device=build_model(2)
params=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.SGD(params,lr=LR,momentum=0.9,weight_decay=0.0005)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
best_loss=float("inf")
for epoch in range(NUM_EPOCHS):
    model.train(); el,t=0.0,time.time()
    for i,(imgs,targets) in enumerate(train_loader):
        imgs=[x.to(device) for x in imgs]
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        losses=sum(model(imgs,targets).values())
        optimizer.zero_grad(); losses.backward(); optimizer.step()
        el+=losses.item()
    scheduler.step(); avg=el/len(train_loader)
    print(f"Epoch {epoch+1} AvgLoss:{avg:.4f} {time.time()-t:.1f}s")
    if avg<best_loss:
        best_loss=avg; torch.save(model.state_dict(),SAVE_BEST)
        print(f"  -> Best saved ({best_loss:.4f})")
torch.save(model.state_dict(),SAVE_FINAL)
print("Training complete.")
