"""Merge dataset_clean, web_clean and web_clean2 into data/dataset_merged.

Existing rows keep their split, so the untouched test set stays untouched.
Web crops are split by video; videos known to show the same person share one split.
"""
import collections
import csv
import os
import random
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "dataset_merged")
RATIOS = (("test", .10), ("val", .20), ("train", .70))
SAME_PERSON = [
    "1WPHI7mamrk GMyGyYBBM2M OetssqWJycg pJ0auP7dbcY 7gEHzfd5WSg FdfgwADHI7k JDjL_QAe3JY SJvslMvn6_I",
    "8xrpnxIfLXI zK5WXahMYDA",
    "20Do_Cse2AI Rc5A-k4SQFc r0uT8AM0z4Y",
]
group_of = {v: i for i, g in enumerate(SAME_PERSON) for v in g.split()}

rows = []
for r in csv.DictReader(open(os.path.join(DATA, "dataset_clean", "labels.csv"), encoding="utf-8")):
    rows.append({"filename": r["filename"], "label": r["thobe"] + r["ghutra"] + r["igal"], "split": r["split"],
                 "origin": "original", "group": r["source"],
                 "path": os.path.join(DATA, "dataset_clean", "images", r["split"], r["filename"])})
for folder, origin in (("web_clean", "web1"), ("web_clean2", "web2")):
    for r in csv.DictReader(open(os.path.join(DATA, folder, "labels.csv"), encoding="utf-8")):
        rows.append({"filename": r["filename"], "label": r["thobe"] + r["ghutra"] + r["igal"], "split": None,
                     "origin": origin, "group": group_of.get(r["video_id"], r["video_id"]),
                     "path": os.path.join(DATA, folder, "images", r["filename"])})
assert len({r["filename"] for r in rows}) == len(rows), "duplicate filename"

groups = collections.defaultdict(list)
for r in rows:
    if r["split"] is None:
        groups[r["group"]].append(r)
freq = collections.Counter(r["label"] for r in rows if r["split"] is None)
strata = collections.defaultdict(list)
for g, items in groups.items():
    strata[min((i["label"] for i in items), key=lambda l: freq[l])].append(g)
rng = random.Random(42)
for label in sorted(strata, key=lambda l: freq[l]):
    gs = sorted(strata[label], key=str)
    rng.shuffle(gs)
    total = sum(len(groups[g]) for g in gs)
    filled = collections.Counter()
    for g in gs:
        # Each group goes to the split furthest below its target for this stratum.
        split = max(RATIOS, key=lambda s: s[1] * total - filled[s[0]])[0]
        filled[split] += len(groups[g])
        for r in groups[g]:
            r["split"] = split

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
for s, _ in RATIOS:
    os.makedirs(os.path.join(OUT, "images", s))
with open(os.path.join(OUT, "labels.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filename", "thobe", "ghutra", "igal", "split", "origin", "group"])
    for r in sorted(rows, key=lambda x: x["filename"]):
        shutil.copy2(r["path"], os.path.join(OUT, "images", r["split"], r["filename"]))
        w.writerow([r["filename"], *r["label"], r["split"], r["origin"], r["group"]])

web = [r for r in rows if r["origin"] != "original"]
gs = collections.defaultdict(set)
for r in web:
    gs[r["group"]].add(r["split"])
print("total", len(rows), "splits", dict(collections.Counter(r["split"] for r in rows)))
print("split x label", sorted(collections.Counter((r["split"], r["label"]) for r in rows).items()))
print("web split x label", sorted(collections.Counter((r["split"], r["label"]) for r in web).items()))
print("web groups in 2 splits", sum(len(v) > 1 for v in gs.values()))
