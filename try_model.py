"""Try the model on an image, a folder of images, or the webcam.

    python try_model.py --source photo.jpg
    python try_model.py --source my_photos_folder
    python try_model.py --source camera        (press q to quit)

Annotated images go to the --out folder. Camera frames are never saved.
"""
import argparse
import os

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["thobe", "ghutra", "igal"]

p = argparse.ArgumentParser()
p.add_argument("--source", required=True, help="image path, folder path, or camera")
p.add_argument("--weights", default=os.path.join(HERE, "models", "trial_best.pt"))
p.add_argument("--yolo", default=os.path.join(HERE, "yolov8n.pt"))
p.add_argument("--out", default=os.path.join(HERE, "results", "try"))
p.add_argument("--threshold", type=float, default=0.5)
p.add_argument("--conf", type=float, default=0.35)
p.add_argument("--camera-id", type=int, default=0)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
clf = models.efficientnet_b0(weights=None)
clf.classifier[1] = nn.Linear(clf.classifier[1].in_features, 3)
clf.load_state_dict(torch.load(a.weights, map_location=dev))
clf = clf.to(dev).eval()
det = YOLO(a.yolo)
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def analyse(frame):
    """Draw results on a BGR frame and return one result per detected person."""
    H, W = frame.shape[:2]
    out = []
    for b in det.predict(frame, classes=[0], conf=a.conf, imgsz=640, device=dev, verbose=False)[0].boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        bw, bh = x2 - x1, y2 - y1
        # Same margins as crop_persons.py, so the classifier sees what it trained on.
        cx1, cy1 = max(0, int(x1 - .10 * bw)), max(0, int(y1 - .15 * bh))
        cx2, cy2 = min(W, int(x2 + .10 * bw)), min(H, int(y2 + .03 * bh))
        crop = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            probs = torch.sigmoid(clf(tf(Image.fromarray(crop)).unsqueeze(0).to(dev)))[0].tolist()
        missing = [n for n, v in zip(NAMES, probs) if v < a.threshold]
        color = (0, 170, 0) if not missing else (0, 0, 220)
        text = "COMPLIANT" if not missing else "MISSING: " + ",".join(missing)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 2)
        cv2.putText(frame, text, (cx1, max(18, cy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out.append((text, probs))
    return out


if a.source == "camera":
    cap = cv2.VideoCapture(a.camera_id)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Try --camera-id 1")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        analyse(frame)
        cv2.imshow("Uniform check - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    paths = ([os.path.join(a.source, f) for f in sorted(os.listdir(a.source)) if f.lower().endswith(exts)]
             if os.path.isdir(a.source) else [a.source])
    os.makedirs(a.out, exist_ok=True)
    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"{path}: could not read image")
            continue
        results = analyse(frame)
        print(f"\n{os.path.basename(path)}: {len(results)} person(s)")
        if not results:
            print("  no person detected")
        for i, (text, pr) in enumerate(results, 1):
            print(f"  person {i}: {text} | thobe={pr[0]:.2f} ghutra={pr[1]:.2f} igal={pr[2]:.2f}")
        cv2.imwrite(os.path.join(a.out, os.path.basename(path)), frame)
    print(f"\nAnnotated images saved to: {a.out}")
