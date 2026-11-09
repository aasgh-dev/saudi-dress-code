"""Apply the manual visual review of the crops, remove near-duplicates, and split.

Input : data/crops_raw/crops/{crops.csv,images}
Output: data/dataset_clean/{labels.csv,removed.csv,changed.csv,images/<split>/}
"""
import collections
import csv
import os
import random
import shutil

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "crops_raw", "crops")
OUT = os.path.join(HERE, "data", "dataset_clean")
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42


def ids(text, suffix):
    return {f"{x}_{suffix}_p0.jpg" for x in text.split()}


REMOVE = {}
def rm(reason, *names):
    for n in names:
        REMOVE[n] = reason


rm("woman", *ids("""1195 1196 1197 1198 1199 1200 1201 1202 1203 1204 1205 1206 1207 1208 1209 1210
    1211 1212 1213 1214 1215 1217 1291 1292 1293 1294 1295 1359""", "sa"))
rm("child", *ids("0531", "e3"), *ids("1141", "sa"), *ids("0412", "e2"),
   *ids("0008 0019 0071 0231", "e1"), *ids("0341", "e2"), *ids("0532 0546 0569 0615", "e3"), *ids("1021 1313", "g7"))
rm("not_a_person", *ids("0294 0304 0393 0404 0409 0428 0439", "e2"), *ids("0682 0721", "e0"), *ids("0010 0027", "e1"))
rm("recolored", *ids("""0006 0014 0016 0023 0026 0032 0035 0038 0044 0061 0075 0091 0099 0101 0107 0116 0140
    0147 0148 0160 0161 0164 0165 0179 0183 0190 0195 0199 0203 0205 0214 0220 0223 0227 0247 0248 0258 0262 0287""", "e1"),
   *ids("0305 0328 0376 0389 0395 0397 0403 0419 0421", "e2"),
   *ids("0463 0471 0474 0490 0499 0500 0501 0502 0524 0537 0538 0547 0548 0562 0625", "e3"), *ids("1268", "g7"))
rm("watermark_or_text", *ids("0043 0047 0104 0119 0120 0127 0159 0278 0280", "e1"), *ids("0678", "e0"),
   *ids("0870", "g2"), *ids("0969 0973 1028 1091 1122 1126", "g7"), *ids("1142 1146 1148 1170 1177 1283", "sa"))
rm("image_artifact", *ids("0418", "e2"), *ids("1042 1121", "g7"))
rm("attire_not_verifiable", *ids("1010 0992 0999 1000 1004 0998 1246 0916 0924 1068 0920 0952 0955", "g7"),
   *ids("0841 0859", "g2"), *ids("0464 0456 0510 0529", "e3"), *ids("1285", "sa"))

RELABEL = {}
for n in ids("0317 0348 0349 0356 0359 0416 0420 0436", "e2") | ids("0764 0768", "e0"):
    RELABEL[n] = ("100", "thobe visible")
for n in ids("1231 1240", "g7"):
    RELABEL[n] = ("111", "igal visible")
RELABEL["0289_e1_p0.jpg"] = ("000", "no thobe")

# Crops the detector flagged as several people. Only those with one clear subject are kept.
REVIEW_KEEP = {
    "0004_e1_p1": "000", "0018_e1_p0": "000", "0058_e1_p0": "000", "0065_e1_p0": "000", "0077_e1_p0": "000",
    "0142_e1_p0": "000", "0156_e1_p0": "000", "0295_e2_p0": "100", "0296_e2_p0": "100", "0296_e2_p1": "100",
    "0370_e2_p0": "100", "0370_e2_p1": "100", "0394_e2_p1": "100", "0448_e3_p0": "111", "0449_e3_p1": "111",
    "0462_e3_p1": "111", "0511_e3_p0": "111", "0511_e3_p1": "100", "0521_e3_p0": "111", "0559_e3_p1": "111",
    "0563_e3_p0": "000", "0563_e3_p1": "111", "0577_e3_p0": "111", "0577_e3_p1": "000", "0592_e3_p0": "111",
    "0594_e3_p0": "000", "0594_e3_p1": "111", "0594_e3_p2": "111", "0610_e3_p0": "111", "0743_e0_p1": "000",
    "0853_g2_p0": "111", "0891_g7_p1": "111", "0898_g7_p0": "111", "0898_g7_p1": "111", "0927_g7_p0": "111",
    "0927_g7_p1": "111", "0927_g7_p2": "111", "1001_g7_p1": "111", "1057_g7_p0": "111", "1057_g7_p1": "111",
    "1260_g7_p0": "111", "1260_g7_p1": "111", "1331_g7_p1": "111",
}


def dhash(im):
    g = np.asarray(im.convert("L").resize((9, 8)), dtype=int)
    return (g[:, 1:] > g[:, :-1]).flatten()


rows = list(csv.DictReader(open(os.path.join(RAW, "crops.csv"), encoding="utf-8-sig")))
known = {r["crop"] for r in rows}
unknown = (set(REMOVE) | set(RELABEL) | {k + ".jpg" for k in REVIEW_KEEP}) - known
assert not unknown, f"ids not in crops.csv: {sorted(unknown)}"

removed, changed, kept = [], [], []
for r in rows:
    n = r["crop"]
    orig = r["thobe"] + r["ghutra"] + r["igal"]
    if n in REMOVE:
        removed.append((n, orig, REMOVE[n]))
    elif r["status"] == "needs_review":
        key = n[:-4]
        if key in REVIEW_KEEP:
            kept.append({"crop": n, "source": r["source"], "label": REVIEW_KEEP[key]})
            changed.append((n, "blank", REVIEW_KEEP[key], "labeled in review"))
        else:
            removed.append((n, "blank", "multi_person_unclear"))
    else:
        label = orig
        if n in RELABEL:
            label, why = RELABEL[n]
            changed.append((n, orig, label, why))
        kept.append({"crop": n, "source": r["source"], "label": label})

# Near-duplicate removal, including mirrored copies. Keeps the first of each group.
for k in kept:
    im = Image.open(os.path.join(RAW, "images", k["crop"])).convert("RGB")
    k["h"], k["hf"] = dhash(im), dhash(ImageOps.mirror(im))
final, conflicts = [], []
for k in kept:
    hit = next((f for f in final if (f["h"] != k["h"]).sum() <= 4 or (f["h"] != k["hf"]).sum() <= 4), None)
    if hit is None:
        final.append(k)
    else:
        removed.append((k["crop"], k["label"], f"duplicate_of:{hit['crop']}"))
        if hit["label"] != k["label"]:
            conflicts.append((k["crop"], k["label"], hit["crop"], hit["label"]))

# Split by source image so crops of one photo never land in two splits.
groups = collections.defaultdict(list)
for k in final:
    groups[k["source"]].append(k)
strata = collections.defaultdict(list)
for src, items in groups.items():
    strata[sorted(i["label"] for i in items)[-1]].append(src)
rng = random.Random(SEED)
for label, srcs in sorted(strata.items()):
    srcs.sort()
    rng.shuffle(srcs)
    n = len(srcs)
    n_test = round(n * SPLITS["test"]) if n >= 10 else (1 if n >= 3 else 0)
    n_val = round(n * SPLITS["val"]) if n >= 5 else (1 if n >= 2 else 0)
    for i, s in enumerate(srcs):
        sp = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for item in groups[s]:
            item["split"] = sp

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
for sp in SPLITS:
    os.makedirs(os.path.join(OUT, "images", sp))
for k in final:
    shutil.copy2(os.path.join(RAW, "images", k["crop"]), os.path.join(OUT, "images", k["split"], k["crop"]))
with open(os.path.join(OUT, "labels.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filename", "thobe", "ghutra", "igal", "split", "source"])
    for k in sorted(final, key=lambda x: x["crop"]):
        w.writerow([k["crop"], *k["label"], k["split"], k["source"]])
with open(os.path.join(OUT, "removed.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["crop", "old_label", "reason"])
    w.writerows(sorted(removed))
with open(os.path.join(OUT, "changed.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["crop", "old_label", "new_label", "reason"])
    w.writerows(sorted(changed))

print("raw", len(rows), "final", len(final), "removed", len(removed), "changed", len(changed))
print("removed by reason", collections.Counter(x[2].split(":")[0] for x in removed).most_common())
print("labels", sorted(collections.Counter(k["label"] for k in final).items()))
print("split x label", sorted(collections.Counter((k["split"], k["label"]) for k in final).items()))
print("label conflicts in duplicates", conflicts)
print("source in 2 splits", sum(len({i["split"] for i in v}) > 1 for v in groups.values()))
print("source mix", {l: collections.Counter(k["crop"].split("_")[1] for k in final if k["label"] == l).most_common() for l in sorted({k["label"] for k in final})})
