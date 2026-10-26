import cv2, torch
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

# 1. Models
yolo = YOLO("yolov8n.pt")

model = models.efficientnet_b0()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
model.load_state_dict(torch.load("uniform_best.pt"))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Camera Loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = yolo(frame, classes=[0], verbose=False)

    for x1, y1, x2, y2 in results[0].boxes.xyxy.int().tolist():
        crop = frame[y1:y2, x1:x2]
        img = tf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0)

        with torch.no_grad():
            thobe, ghutra, igal = torch.sigmoid(model(img))[0]

        is_saudi = thobe > 0.5 or (ghutra > 0.5 and igal > 0.4)
        color = (0, 255, 0) if is_saudi else (0, 0, 255)
        text = "Saudi Dress" if is_saudi else "Not Saudi Dress"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
