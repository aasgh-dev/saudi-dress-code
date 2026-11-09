"""Apply the manual review of the video crops. Copies a capped, labeled selection to data/web_clean.

Not merged with dataset_clean and not used for training yet.
Labels are [thobe, ghutra, igal]. Capped per person/setting because frames of one person are near-copies.
"""
import collections
import csv
import os
import shutil

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "web_raw", "crops")
OUT = os.path.join(HERE, "data", "web_clean")

KEEP = {
    "110": """UVQOz88OXFQ_00020_p0 UVQOz88OXFQ_00120_p0 UVQOz88OXFQ_00160_p0""",
    "011": "XBSxeG6jfbw_00040_p0 XBSxeG6jfbw_00340_p0",
    "111": """3CKGrixtCJU_00160_p0 3CKGrixtCJU_00620_p0 3CKGrixtCJU_01840_p1 3CKGrixtCJU_02140_p0 3CKGrixtCJU_03060_p0
        3CKGrixtCJU_03660_p0 3CKGrixtCJU_04420_p0 3CKGrixtCJU_05020_p0 3CKGrixtCJU_05480_p1
        -VfyaWDPR80_00020_p1 -VfyaWDPR80_00040_p0 -VfyaWDPR80_00060_p0 4kb2VK1yxNU_00960_p0
        8RkNANMKY-k_00020_p0 8RkNANMKY-k_00620_p0 8RkNANMKY-k_02140_p0 8RkNANMKY-k_03660_p0
        8y-FA2anhUE_00080_p3 8y-FA2anhUE_00640_p0 8y-FA2anhUE_01340_p1 8y-FA2anhUE_01340_p2 8y-FA2anhUE_01600_p0 8y-FA2anhUE_02240_p0 8y-FA2anhUE_02300_p0 8y-FA2anhUE_02360_p0
        AuZNSZq_aVI_00020_p0 AuZNSZq_aVI_00080_p0 AuZNSZq_aVI_00300_p0 AuZNSZq_aVI_00440_p0 AuZNSZq_aVI_00560_p0
        CK15I4vHFhc_00040_p0 CK15I4vHFhc_00080_p0 FUBWyANnwgE_00140_p0 FUBWyANnwgE_00360_p0 FUBWyANnwgE_01800_p0 FUBWyANnwgE_01940_p0
        GMyGyYBBM2M_00020_p0 GMyGyYBBM2M_00720_p0 GMyGyYBBM2M_02500_p0 LzbfMq6OS2U_00200_p0 LzbfMq6OS2U_02200_p0 LzbfMq6OS2U_03860_p0 LzbfMq6OS2U_05680_p0 LzbfMq6OS2U_06040_p0
        OetssqWJycg_00020_p0 OetssqWJycg_00800_p0 OetssqWJycg_03400_p0 OetssqWJycg_06540_p0 SJvslMvn6_I_00400_p0 SJvslMvn6_I_02920_p0 SJvslMvn6_I_03900_p0
        fBLjs-VkBKE_00240_p0 fBLjs-VkBKE_02780_p0 fBLjs-VkBKE_05080_p0 fBLjs-VkBKE_07600_p0 fllo3KpIcSw_00040_p0 ha9wvwwOMpM_00040_p0
        pJ0auP7dbcY_00560_p0 pJ0auP7dbcY_03300_p0 pJ0auP7dbcY_04680_p0 pJ0auP7dbcY_07700_p0 wfje9jXUx4E_00020_p0 wfje9jXUx4E_00140_p0 wfje9jXUx4E_00280_p0
        xv7Bqjs0YEM_00700_p1 xv7Bqjs0YEM_00800_p0 xv7Bqjs0YEM_01320_p2""",
    "100": """1WPHI7mamrk_00020_p0 1WPHI7mamrk_00560_p0 1WPHI7mamrk_01240_p0 1WPHI7mamrk_02200_p0 1WPHI7mamrk_00140_p0 1WPHI7mamrk_00280_p0 1WPHI7mamrk_01640_p0 1WPHI7mamrk_05340_p0
        4kb2VK1yxNU_00120_p0 4kb2VK1yxNU_00160_p0 4kb2VK1yxNU_00260_p0 4kb2VK1yxNU_00300_p0 4kb2VK1yxNU_00440_p0 4kb2VK1yxNU_01160_p1
        8RkNANMKY-k_00120_p0 8RkNANMKY-k_00520_p0 8RkNANMKY-k_02240_p0 8RkNANMKY-k_02960_p0
        8y-FA2anhUE_00080_p2 8y-FA2anhUE_01860_p0 8y-FA2anhUE_01400_p0 8y-FA2anhUE_01540_p0 8y-FA2anhUE_01660_p2 8y-FA2anhUE_02160_p1 8y-FA2anhUE_01480_p4
        FUBWyANnwgE_01260_p1 FUBWyANnwgE_01880_p2 GMyGyYBBM2M_00360_p0 GMyGyYBBM2M_03580_p0 OetssqWJycg_02360_p0 OetssqWJycg_09680_p0
        SJvslMvn6_I_00200_p0 SJvslMvn6_I_01360_p0 SJvslMvn6_I_05640_p0 SJvslMvn6_I_07000_p0 Z88rCA_B-eI_00020_p0 Z88rCA_B-eI_00040_p0 Z88rCA_B-eI_00080_p0 Z88rCA_B-eI_00120_p0
        pJ0auP7dbcY_06320_p0 pJ0auP7dbcY_07980_p0 xv7Bqjs0YEM_00700_p3 xv7Bqjs0YEM_01320_p0 xv7Bqjs0YEM_01320_p3 xv7Bqjs0YEM_01840_p0 xv7Bqjs0YEM_02000_p0""",
    "000": """0LKjtOswIjU_00020_p0 0LKjtOswIjU_00260_p0 3CKGrixtCJU_00020_p0 3CKGrixtCJU_00160_p1 3CKGrixtCJU_01840_p0 3CKGrixtCJU_05020_p1
        4kb2VK1yxNU_00180_p1 4kb2VK1yxNU_00260_p1 4kb2VK1yxNU_00500_p0 4kb2VK1yxNU_00500_p2 4kb2VK1yxNU_00740_p0
        4maJM57oo3M_00040_p0 4maJM57oo3M_00060_p0 4maJM57oo3M_00100_p0 BbgaSThZeUs_00020_p0 BbgaSThZeUs_00180_p0
        E1R86snqawo_00260_p2 E1R86snqawo_00280_p1 E1R86snqawo_00300_p2 E1R86snqawo_00320_p0 E1R86snqawo_00380_p0 E1R86snqawo_00400_p3
        FUBWyANnwgE_00500_p0 FUBWyANnwgE_00840_p0 FUBWyANnwgE_02020_p0""",
}

manifest = {r["crop"]: r for r in csv.DictReader(open(os.path.join(HERE, "data", "web_raw", "manifest.csv"), encoding="utf-8"))}
items = [(n + ".jpg", l) for l, s in KEEP.items() for n in s.split()]
missing = [n for n, _ in items if n not in manifest]
assert not missing, missing
assert len({n for n, _ in items}) == len(items), "a crop has two labels"

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, "images"))
with open(os.path.join(OUT, "labels.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["filename", "thobe", "ghutra", "igal", "video_id", "url", "second"])
    for n, l in sorted(items):
        shutil.copy2(os.path.join(RAW, n), os.path.join(OUT, "images", n))
        m = manifest[n]
        w.writerow([n, *l, m["video_id"], m["url"], m["second"]])

C, W, H = 8, 150, 190
for label in sorted(KEEP):
    ns = sorted(n for n, l in items if l == label)
    rows = (len(ns) + C - 1) // C
    S = Image.new("RGB", (C * W, rows * (H + 14)), "white")
    D = ImageDraw.Draw(S)
    for i, n in enumerate(ns):
        im = Image.open(os.path.join(RAW, n)).convert("RGB")
        im.thumbnail((W - 4, H))
        x, y = i % C * W, i // C * (H + 14)
        S.paste(im, (x, y + 14))
        D.text((x + 2, y + 1), n[:-4][:24], fill="black")
    S.save(os.path.join(OUT, f"review_{label}.jpg"), quality=90)

print("kept", len(items), "of", len(manifest))
print("labels", sorted(collections.Counter(l for _, l in items).items()))
print("videos per label", {l: len({n[:11] for n, x in items if x == l}) for l in sorted(KEEP)})
