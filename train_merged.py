"""Train on data/dataset_merged. Evaluates on val only; the test split is never loaded.

Rare label combinations are sampled more often. The best epoch is chosen by mean balanced accuracy across
components, because F1 on the present class would hide a model that misses absent items.
"""
import collections
import csv
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "dataset_merged")
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
ROWS = list(csv.DictReader(open(os.path.join(DATA, "labels.csv"), encoding="utf-8")))


class DS(Dataset):
    def __init__(self, split, tf):
        self.rows, self.split, self.tf = [r for r in ROWS if r["split"] == split], split, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(os.path.join(DATA, "images", self.split, r["filename"])).convert("RGB")
        return self.tf(im), torch.tensor([float(r[k]) for k in LABELS]), r["filename"]


def combo(r):
    return r["thobe"] + r["ghutra"] + r["igal"]


def predict(model, loader, dev):
    model.eval()
    ps, ys, ns = [], [], []
    with torch.no_grad():
        for x, y, n in loader:
            ps.append(torch.sigmoid(model(x.to(dev))).cpu())
            ys.append(y)
            ns.extend(n)
    return torch.cat(ps), torch.cat(ys), ns


def stats(h, y):
    out = {}
    for i, k in enumerate(LABELS):
        tp = int((h[:, i] * y[:, i]).sum()); fp = int((h[:, i] * (1 - y[:, i])).sum())
        fn = int(((1 - h[:, i]) * y[:, i]).sum()); tn = int(((1 - h[:, i]) * (1 - y[:, i])).sum())
        out[k] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(tp / max(tp + fp, 1), 3),
                  "recall": round(tp / max(tp + fn, 1), 3), "absent_detected": round(tn / max(tn + fp, 1), 3),
                  "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3)}
    return out


def balanced(h, y):
    s = stats(h, y)
    return sum((v["recall"] + v["absent_detected"]) / 2 for v in s.values()) / 3


def report(p, y, names):
    h = (p > 0.5).float()
    comp_t, comp_p = y.all(1), h.all(1)
    by = collections.defaultdict(lambda: [0, 0])
    for yi, hi in zip(y.tolist(), h.tolist()):
        k = "".join(str(int(v)) for v in yi)
        by[k][1] += 1
        by[k][0] += yi == hi
    return {"n": len(y), "per_component": stats(h, y), "exact_by_true_label": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())},
            "compliance_accuracy": round((comp_t == comp_p).float().mean().item(), 3),
            "non_compliant_wrongly_accepted": f"{int((~comp_t & comp_p).sum())}/{int((~comp_t).sum())}",
            "compliant_wrongly_rejected": f"{int((comp_t & ~comp_p).sum())}/{int(comp_t.sum())}"}


def main():
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", flush=True)
    os.makedirs(OUT, exist_ok=True)
    tr_ds, va_ds = DS("train", TF_TRAIN), DS("val", TF_EVAL)
    freq = collections.Counter(combo(r) for r in tr_ds.rows)
    weights = [1 / freq[combo(r)] ** 0.5 for r in tr_ds.rows]
    tr = DataLoader(tr_ds, batch_size=BATCH, sampler=WeightedRandomSampler(weights, len(weights), True),
                    num_workers=2, persistent_workers=True)
    va = DataLoader(va_ds, batch_size=64, num_workers=2, persistent_workers=True)
    print("train", len(tr_ds), sorted(freq.items()), "val", len(va_ds), flush=True)

    m = models.efficientnet_b0(weights="IMAGENET1K_V1")
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    path = os.path.join(OUT, "merged_best.pt")
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
        b = balanced((p > 0.5).float(), y)
        print(f"epoch {ep:2d} loss={total / len(tr_ds):.4f} val_balanced={b:.3f}", flush=True)
        if b > best:
            best, best_ep = b, ep
            torch.save(m.state_dict(), path)

    m.load_state_dict(torch.load(path, map_location=dev))
    p, y, names = predict(m, va, dev)
    origin = {r["filename"]: r["origin"] for r in ROWS}
    web = torch.tensor([origin[n] != "original" for n in names])
    res = {"best_epoch": best_ep, "val_all": report(p, y, names), "val_web_only": report(p[web], y[web], [n for n, w in zip(names, web) if w]),
           "val_original_only": report(p[~web], y[~web], [n for n, w in zip(names, web) if not w])}
    json.dump(res, open(os.path.join(OUT, "merged_val_results.json"), "w"), indent=2)
    h = (p > 0.5).float()
    with open(os.path.join(OUT, "merged_val_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "origin", "true", "pred", "p_thobe", "p_ghutra", "p_igal"])
        for n, pi, yi, hi in zip(names, p.tolist(), y.tolist(), h.tolist()):
            w.writerow([n, origin[n], "".join(str(int(v)) for v in yi), "".join(str(int(v)) for v in hi), *[round(v, 3) for v in pi]])
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
