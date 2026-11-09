"""Train the multi-label attire classifier on data/dataset_clean. Evaluates on val only."""
import collections
import csv
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "dataset_clean")
OUT = os.path.join(HERE, "models")
LABELS = ["thobe", "ghutra", "igal"]
EPOCHS, BATCH, LR, SEED = 20, 32, 3e-4, 42

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.0),
    transforms.ToTensor(), norm])
TF_EVAL = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])


class DS(Dataset):
    def __init__(self, split, tf):
        rows = list(csv.DictReader(open(os.path.join(DATA, "labels.csv"), encoding="utf-8")))
        self.rows, self.split, self.tf = [r for r in rows if r["split"] == split], split, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(os.path.join(DATA, "images", self.split, r["filename"])).convert("RGB")
        return self.tf(im), torch.tensor([float(r[k]) for k in LABELS]), r["filename"]


def predict(model, loader, dev):
    model.eval()
    probs, ys, names = [], [], []
    with torch.no_grad():
        for x, y, n in loader:
            probs.append(torch.sigmoid(model(x.to(dev))).cpu())
            ys.append(y)
            names.extend(n)
    return torch.cat(probs), torch.cat(ys), names


def f1s(p, y):
    h = (p > 0.5).float()
    tp, fp, fn = (h * y).sum(0), (h * (1 - y)).sum(0), ((1 - h) * y).sum(0)
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def main():
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", flush=True)
    os.makedirs(OUT, exist_ok=True)
    tr = DataLoader(DS("train", TF_TRAIN), batch_size=BATCH, shuffle=True, num_workers=2, persistent_workers=True)
    va = DataLoader(DS("val", TF_EVAL), batch_size=64, num_workers=2, persistent_workers=True)

    m = models.efficientnet_b0(weights="IMAGENET1K_V1")
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    path = os.path.join(OUT, "trial_best.pt")
    best, best_ep = -1, 0
    for ep in range(1, EPOCHS + 1):
        m.train()
        total = 0
        for x, y, _ in tr:
            opt.zero_grad()
            loss = loss_fn(m(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        p, y, _ = predict(m, va, dev)
        f = f1s(p, y)
        print(f"epoch {ep:2d} loss={total / len(tr.dataset):.4f} f1 thobe={f[0]:.3f} ghutra={f[1]:.3f} igal={f[2]:.3f}", flush=True)
        if f.mean().item() > best:
            best, best_ep = f.mean().item(), ep
            torch.save(m.state_dict(), path)

    m.load_state_dict(torch.load(path, map_location=dev))
    p, y, names = predict(m, va, dev)
    h = (p > 0.5).float()
    res = {"best_epoch": best_ep, "n_val": len(y), "per_component": {}}
    for i, k in enumerate(LABELS):
        tp = int((h[:, i] * y[:, i]).sum()); fp = int((h[:, i] * (1 - y[:, i])).sum())
        fn = int(((1 - h[:, i]) * y[:, i]).sum()); tn = int(((1 - h[:, i]) * (1 - y[:, i])).sum())
        res["per_component"][k] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": round((tp + tn) / len(y), 3),
                                   "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                                   "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3)}
    combos = collections.defaultdict(lambda: [0, 0])
    rows = []
    for n, pi, yi, hi in zip(names, p.tolist(), y.tolist(), h.tolist()):
        t = "".join(str(int(v)) for v in yi)
        combos[t][1] += 1
        combos[t][0] += yi == hi
        rows.append([n, t, "".join(str(int(v)) for v in hi), *[round(v, 3) for v in pi]])
    res["exact_match_by_true_label"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(combos.items())}
    comp_t, comp_p = y.all(1), h.all(1)
    res["compliance"] = {"accuracy": round((comp_t == comp_p).float().mean().item(), 3),
                         "non_compliant_wrongly_accepted": f"{int((~comp_t & comp_p).sum())}/{int((~comp_t).sum())}",
                         "compliant_wrongly_rejected": f"{int((comp_t & ~comp_p).sum())}/{int(comp_t.sum())}"}
    json.dump(res, open(os.path.join(OUT, "trial_val_results.json"), "w"), indent=2)
    with open(os.path.join(OUT, "trial_val_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true", "pred", "p_thobe", "p_ghutra", "p_igal"])
        w.writerows(rows)
    print(json.dumps(res, indent=2))
    print("errors:")
    for r in rows:
        if r[1] != r[2]:
            print(r)


if __name__ == "__main__":
    main()
