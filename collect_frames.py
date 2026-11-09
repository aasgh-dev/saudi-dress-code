"""Sample frames from videos, crop people with YOLO, drop near-duplicates per video.

Videos are streamed, not saved. Output: data/web_raw/{crops/,manifest.csv}. Nothing here is labeled or used for training.
"""
import csv
import os
import sys

import cv2
import numpy as np
import yt_dlp
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "web_raw")
IDS = """E3R6k6fGvw0 Z88rCA_B-eI BbgaSThZeUs 4kb2VK1yxNU 4maJM57oo3M FUBWyANnwgE E1R86snqawo 2LVfxJjBmsI
OetssqWJycg 3CKGrixtCJU LzbfMq6OS2U GMyGyYBBM2M AHS_lF1csgc pJ0auP7dbcY 1WPHI7mamrk SJvslMvn6_I
fBLjs-VkBKE 8RkNANMKY-k UVQOz88OXFQ CK15I4vHFhc fllo3KpIcSw wfje9jXUx4E
XBSxeG6jfbw xv7Bqjs0YEM 0LKjtOswIjU AuZNSZq_aVI -VfyaWDPR80 ha9wvwwOMpM 8y-FA2anhUE""".split()
STEP, MAX_FRAMES, CONF, MIN_AREA = 20, 40, 0.5, 0.08

os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)
det = YOLO(os.path.join(HERE, "yolov8n.pt"))
ydl = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": "bestvideo[height<=720][vcodec^=avc1][protocol=https]/bestvideo[height<=720][vcodec^=avc1]"})


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8)).astype(int)
    return (g[:, 1:] > g[:, :-1]).flatten()


with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["crop", "video_id", "url", "second", "conf"])
    for vid in IDS:
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            info = ydl.extract_info(url, download=False)
            cap = cv2.VideoCapture(info["url"])
            dur = info.get("duration") or 0
            times = list(range(STEP, dur, STEP))
            if len(times) > MAX_FRAMES:
                times = [times[i * len(times) // MAX_FRAMES] for i in range(MAX_FRAMES)]
            seen, saved = [], 0
            for t in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, fr = cap.read()
                if not ok:
                    continue
                H, W = fr.shape[:2]
                for i, b in enumerate(det.predict(fr, classes=[0], conf=CONF, imgsz=640, verbose=False)[0].boxes):
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    bw, bh = x2 - x1, y2 - y1
                    if bw * bh < MIN_AREA * W * H:
                        continue
                    c = fr[max(0, int(y1 - .15 * bh)):min(H, int(y2 + .03 * bh)), max(0, int(x1 - .1 * bw)):min(W, int(x2 + .1 * bw))]
                    h = dhash(c)
                    if any((h != s).sum() <= 8 for s in seen):
                        continue
                    seen.append(h)
                    name = f"{vid}_{t:05d}_p{i}.jpg"
                    cv2.imwrite(os.path.join(OUT, "crops", name), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    w.writerow([name, vid, url, t, round(float(b.conf[0]), 3)])
                    saved += 1
            cap.release()
            f.flush()
            print(vid, "duration", dur, "crops", saved, flush=True)
        except Exception as e:
            print(vid, "FAILED", str(e)[:150], flush=True)
print("done", file=sys.stdout)
