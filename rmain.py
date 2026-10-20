import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

photo = cv2.imread("group_test.jpg")
H, W = photo.shape[:2]

result = model(photo, classes=[0], conf=0.5)   # 0 = person

for i, (x1, y1, x2, y2) in enumerate(result[0].boxes.xyxy.int().tolist()):
    # هامش 5% عشان ما ينقص الشماغ من فوق
    dx, dy = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
    crop = photo[max(0, y1 - dy):min(H, y2 + dy), max(0, x1 - dx):min(W, x2 + dx)]

    cv2.imwrite(f"person_{i+1}.jpg", crop)
    cv2.imshow(f"person {i+1}", crop)

cv2.imshow("all", result[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()