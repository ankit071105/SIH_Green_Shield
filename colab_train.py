# colab_train.py
# Usage (in Colab): python colab_train.py --data_dir /content/dataset --epochs 12 --bs 32 --lr 3e-4 --out /content/best_model.pt --save_to_drive
import argparse, os, shutil, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--epochs', type=int, default=12)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--out', type=str, default='models/best_model.pt')
parser.add_argument('--save_to_drive', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

train_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

root = Path(args.data_dir)
train_ds = datasets.ImageFolder(root/'train', transform=train_tf)
val_ds = datasets.ImageFolder(root/'val', transform=val_tf)
print('Classes:', train_ds.classes)
CLASSES = train_ds.classes

train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

best_acc = 0.0
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler() if device.type == 'cuda' else None

for epoch in range(1, args.epochs+1):
    model.train()
    start = time.time()
    total, correct, loss_sum = 0,0,0.0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    train_acc = correct/total if total>0 else 0.0

    # validation
    model.eval()
    v_total, v_correct = 0,0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            v_correct += (logits.argmax(1)==y).sum().item()
            v_total += y.size(0)
    val_acc = v_correct/v_total if v_total>0 else 0.0
    scheduler.step()
    print(f'Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}')
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), out_path)
        print('Saved best model to', out_path)

print('Training completed. Best val acc:', best_acc)
if args.save_to_drive:
    try:
        drive_path = '/content/drive/MyDrive/sih_models'
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy(str(out_path), drive_path)
        print('Copied best model to Drive:', drive_path)
    except Exception as e:
        print('Could not copy to Drive:', e)
