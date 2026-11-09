"""Collect candidate no-igal crops from targeted video searches. Videos are streamed, not saved.

Output: data/web_raw_110/{crops/,manifest.csv,zoom_*.jpg}. Candidates only, nothing is labeled here.
"""
import csv
import os
import sys

import cv2
import numpy as np
import yt_dlp
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "web_raw_110")
QUERIES = ["لقاء مع الشيخ السعودي", "محاضرة شيخ سعودي", "برنامج فتاوى السعودية", "درس علمي الشيخ مسجد الرياض",
           "دكتور جامعة سعودي لقاء تلفزيوني", "طريقة لف الشماغ بدون عقال", "لبس الغترة بدون عقال", "شاب سعودي شماغ بدون عقال سناب",
           "مقابلة مسؤول سعودي غترة", "خطبة الجمعة السعودية", "لقاء كبار السن السعودية مزرعة", "رجل سعودي يتكلم عن تجربته بودكاست"]
PER_QUERY, FRAMES, CONF, MIN_AREA = 12, 8, 0.5, 0.12
MAX_SECONDS = 2 * 3600

# Second round: real-life footage resembling phone photos, seated people, night scenes, several people side by side.
if len(sys.argv) > 1 and sys.argv[1] == "real":
    OUT = os.path.join(HERE, "data", "web_raw_real")
    QUERIES = ["فلوق شباب سعوديين استراحة", "سوالف شباب سعوديين جلسة ليل", "فلوق يومي الرياض ثوب", "جلسة بر سعوديين شتاء",
               "لقطات جوال سعوديين في المقهى", "فلوق جدة شباب ليل", "اجتماع عائلي رجال مجلس سعودي", "زيارة سوق شعبي السعودية فلوق",
               "تخرج جامعة سعودية طلاب ثوب", "طلاب جامعة سعوديين لقاء الحرم الجامعي", "موظفين سعوديين مكتب فلوق", "سناب شاب سعودي يومياتي"]

done = set()
for folder in ("web_raw", "web_raw_110"):
    done |= {r["video_id"] for r in csv.DictReader(open(os.path.join(HERE, "data", folder, "manifest.csv"), encoding="utf-8"))}
os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)
det = YOLO(os.path.join(HERE, "yolov8n.pt"))
flat = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True})
ydl = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": "bestvideo[height<=720][vcodec^=avc1][protocol=https]/bestvideo[height<=720][vcodec^=avc1]"})

ids = []
for q in QUERIES:
    try:
        for e in flat.extract_info(f"ytsearch{PER_QUERY}:{q}", download=False)["entries"]:
            if e["id"] not in done and e["id"] not in ids and (e.get("duration") or 0) >= 60:
                ids.append(e["id"])
    except Exception as ex:
        print("SEARCH FAILED", q, str(ex)[:100], flush=True)
print("videos", len(ids), flush=True)


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8)).astype(int)
    return (g[:, 1:] > g[:, :-1]).flatten()


with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["crop", "video_id", "url", "second", "conf"])
    for vid in ids:
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            info = ydl.extract_info(url, download=False)
            dur = min(info.get("duration") or 0, MAX_SECONDS)
            cap = cv2.VideoCapture(info["url"])
            seen, saved = [], 0
            for k in range(FRAMES):
                t = int(dur * (k + 1) / (FRAMES + 1))
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, fr = cap.read()
                if not ok:
                    continue
                H, W = fr.shape[:2]
                boxes = det.predict(fr, classes=[0], conf=CONF, imgsz=640, verbose=False)[0].boxes
                for i, b in enumerate(boxes):
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    bw, bh = x2 - x1, y2 - y1
                    if bw * bh < MIN_AREA * W * H:
                        continue
                    c = fr[max(0, int(y1 - .15 * bh)):min(H, int(y2 + .03 * bh)), max(0, int(x1 - .1 * bw)):min(W, int(x2 + .1 * bw))]
                    h = dhash(c)
                    if any((h != s).sum() <= 10 for s in seen):
                        continue
                    seen.append(h)
                    name = f"{vid}_{t:05d}_p{i}.jpg"
                    cv2.imwrite(os.path.join(OUT, "crops", name), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    w.writerow([name, vid, url, t, round(float(b.conf[0]), 3)])
                    saved += 1
            cap.release()
            f.flush()
            print(vid, "crops", saved, flush=True)
        except Exception as ex:
            print(vid, "FAILED", str(ex)[:120], flush=True)

# Review sheets: full crop beside an enlarged head region, so a missing igal can be judged.
names = sorted(os.listdir(os.path.join(OUT, "crops")))
C, R, W_, H_ = 6, 6, 200, 170
for s in range(0, len(names), C * R):
    sheet = np.full((R * (H_ + 16), C * W_, 3), 255, np.uint8)
    for i, n in enumerate(names[s:s + C * R]):
        im = cv2.imread(os.path.join(OUT, "crops", n))
        ih, iw = im.shape[:2]
        side = min(iw, int(ih * .6))
        l = max(0, (iw - side) // 2)
        tile = cv2.resize(im[0:int(side * .85), l:l + side], (W_ - 2, H_))
        x, y = i % C * W_, i // C * (H_ + 16)
        sheet[y + 16:y + 16 + H_, x:x + W_ - 2] = tile
        cv2.putText(sheet, n[:-4][:26], (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(OUT, f"zoom_{s // (C * R):03d}.jpg"), sheet)
print("total crops", len(names), "sheets", (len(names) + C * R - 1) // (C * R))
