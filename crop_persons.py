"""Crop people from the dataset with YOLO and carry over their labels.

Usage (local or Colab):
    python crop_persons.py --images saudi_attire_clean/images \
        --labels saudi_attire_clean/labels.csv --out crops

Output: <out>/images/*.jpg and <out>/crops.csv
Status values in crops.csv:
    ok            one main person; label copied from the source image, still review it
    needs_review  several similar-sized people; label is blank, fill it in by hand
Images with no detected person are listed in <out>/no_person.csv.
"""
import argparse
import csv
import os

from PIL import Image
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--images", required=True)
p.add_argument("--labels", required=True)
p.add_argument("--out", default="crops")
p.add_argument("--weights", default="yolov8n.pt")
p.add_argument("--conf", type=float, default=0.35, help="minimum detection confidence")
p.add_argument("--min-area", type=float, default=0.05, help="ignore boxes smaller than this share of the image")
p.add_argument("--main-ratio", type=float, default=0.30, help="others below this share of the largest box count as background")
p.add_argument("--top", type=float, default=0.15, help="extra margin above the box, as a share of box height")
p.add_argument("--side", type=float, default=0.10, help="extra margin left and right, as a share of box width")
p.add_argument("--bottom", type=float, default=0.03, help="extra margin below the box, as a share of box height")
p.add_argument("--limit", type=int, default=0, help="process only the first N images, for testing")
a = p.parse_args()

os.makedirs(f"{a.out}/images", exist_ok=True)
rows = list(csv.DictReader(open(a.labels, encoding="utf-8-sig")))
if a.limit:
    rows = rows[: a.limit]
model = YOLO(a.weights)
FIELDS = ["crop", "source", "thobe", "ghutra", "igal", "status", "conf"]
counts = {"ok": 0, "needs_review": 0, "no_person": 0}

with open(f"{a.out}/crops.csv", "w", newline="", encoding="utf-8") as fc, \
        open(f"{a.out}/no_person.csv", "w", newline="", encoding="utf-8") as fn:
    wc = csv.DictWriter(fc, FIELDS)
    wc.writeheader()
    wn = csv.writer(fn)
    wn.writerow(["source", "thobe", "ghutra", "igal"])
    for r in rows:
        path = os.path.join(a.images, r["filename"])
        im = Image.open(path).convert("RGB")
        W, H = im.size
        res = model.predict(im, classes=[0], conf=a.conf, imgsz=640, verbose=False)[0]
        boxes = [(*map(float, b.xyxy[0].tolist()), float(b.conf[0])) for b in res.boxes]
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        boxes = sorted((b for b in boxes if area(b) >= a.min_area * W * H), key=area, reverse=True)
        if not boxes:
            wn.writerow([r["filename"], r["thobe"], r["ghutra"], r["igal"]])
            counts["no_person"] += 1
            continue
        # Background people are dropped; the image label describes the main person.
        boxes = [b for b in boxes if area(b) >= a.main_ratio * area(boxes[0])]
        status = "ok" if len(boxes) == 1 else "needs_review"
        stem = os.path.splitext(r["filename"])[0]
        for i, (x1, y1, x2, y2, c) in enumerate(boxes):
            bw, bh = x2 - x1, y2 - y1
            box = (max(0, int(x1 - a.side * bw)), max(0, int(y1 - a.top * bh)),
                   min(W, int(x2 + a.side * bw)), min(H, int(y2 + a.bottom * bh)))
            name = f"{stem}_p{i}.jpg"
            im.crop(box).save(f"{a.out}/images/{name}", quality=95)
            keep = status == "ok"
            wc.writerow({"crop": name, "source": r["filename"], "thobe": r["thobe"] if keep else "",
                         "ghutra": r["ghutra"] if keep else "", "igal": r["igal"] if keep else "",
                         "status": status, "conf": round(c, 3)})
            counts[status] += 1
print(counts)
