import cv2
import os
import csv
import numpy as np
from ultralytics import YOLO


VIDEO_SOURCE = "guys.mp4"
MODEL_PATH = "yolov8n.pt"

CONF = 0.4
TARGET_COLOR = "orange"
SCALE = 0.3


CLASSES_TO_TRACK = [0, 24, 26, 28, 39, 41]

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "color_objects.csv")

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)
csv_rows = []



def get_dominant_color(bgr_patch):
    """Повертає домінуючий колір: red, orange, yellow, green, blue, purple, other"""
    if bgr_patch.size == 0:
        return "other"

    hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_patch)


    mask = (s > 50) & (v > 50)
    h = h[mask]

    if len(h) == 0:
        return "other"


    hist, bins = np.histogram(h, bins=180, range=(0, 180))
    dominant_hue = np.argmax(hist)


    if (dominant_hue < 10 or dominant_hue > 160):
        return "red"
    elif 10 <= dominant_hue < 25:
        return "orange"
    elif 25 <= dominant_hue < 35:
        return "yellow"
    elif 35 <= dominant_hue < 85:
        return "green"
    elif 85 <= dominant_hue < 130:
        return "blue"
    elif 130 <= dominant_hue < 160:
        return "purple"
    else:
        return "other"



print("Запуск детекції. Натисніть ESC для виходу.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    results = model(frame, conf=CONF, verbose=False)[0]

    if results.boxes is not None:
        for box_data in results.boxes:
            cls = int(box_data.cls[0])
            if cls not in CLASSES_TO_TRACK:
                continue

            x1, y1, x2, y2 = map(int, box_data.xyxy[0].cpu().numpy())
            obj_patch = frame[y1:y2, x1:x2]
            obj_color = get_dominant_color(obj_patch)


            if obj_color == TARGET_COLOR:
                color_bgr = (0, 0, 255)
            else:
                color_bgr = (0, 255, 0)


            if cls == 0:
                label = f"Person: {obj_color}"
            else:
                label = f"{model.names[cls]}: {obj_color}"


            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)


            csv_rows.append([current_time_sec, model.names[cls], obj_color, x1, y1, x2, y2])


    frame_resized = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    cv2.imshow("Object Color Detector", frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time(s)", "Class", "Color", "x1", "y1", "x2", "y2"])
    writer.writerows(csv_rows)

print(f"Результати збережено в: {CSV_PATH}")
