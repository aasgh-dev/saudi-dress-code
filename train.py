import csv, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

LABELS = ["thobe", "ghutra", "igal"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), norm
])
tf_val = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

class DS(Dataset):
    def __init__(self, split, tf):
        rows = list(csv.DictReader(open("labels.csv")))
        self.rows = [r for r in rows if r["split"] == split]
        self.tf = tf
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tf(Image.open(f"images/{r['filename']}").convert("RGB"))
        return x, torch.tensor([float(r[k]) for k in LABELS])

tr = DataLoader(DS("train", tf_train), batch_size=32, shuffle=True)
va = DataLoader(DS("val", tf_val), batch_size=64)

m = models.efficientnet_b0(weights="IMAGENET1K_V1")
m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
m = m.to(dev)

opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
loss_fn = nn.BCEWithLogitsLoss()

best = 0
for ep in range(20):
    m.train()
    for x, y in tr:
        opt.zero_grad()
        loss = loss_fn(m(x.to(dev)), y.to(dev))
        loss.backward()
        opt.step()

    m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in va:
            p = (torch.sigmoid(m(x.to(dev))) > 0.5).float()
            correct += (p == y.to(dev)).all(dim=1).sum().item()
            total += len(y)

    acc = correct / total
    print(f"epoch {ep+1} loss={loss.item():.3f} val_acc={acc:.2f}")
    if acc > best:
        best = acc
        torch.save(m.state_dict(), "uniform_best.pt")
