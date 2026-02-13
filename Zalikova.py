import cv2
import os
from ultralytics import YOLO
import subprocess
import time
import csv

PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_DIR = os.path.join(OUTPUT_DIR, 'videos')
INPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'video.mp4')
OUTPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'output_video.mp4')
CSV_PATH = os.path.join(OUTPUT_DIR, 'vehicle_speeds.csv')

USE_WEBCAM = False
source = 0 if USE_WEBCAM else "https://www.youtube.com/watch?v=Lxqcg1qt0XU"

MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.5
TRACKER = "bytetrack.yaml"
SAVE_VIDEO = True
model = YOLO(MODEL_PATH)

if isinstance(source, str) and "youtube.com" in source:
    cmd = ["streamlink", "--stream-url", source, "best"]
    stream_url = subprocess.check_output(cmd).decode().strip()
    cap = cv2.VideoCapture(stream_url)
else:
    cap = cv2.VideoCapture(source)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 30

writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

seen_id_total = set()
seen_id_class = {}
Y1 = 400
Y2 = 600
REAL_DISTANCE_M = 10.0
cross_times = {}
csv_data = []

# Vehicle classes: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(frame, classes=VEHICLE_CLASSES, conf=CONF_THRESH, tracker=TRACKER, persist=True, verbose=False)
    r = result[0]

    cv2.line(frame, (0, Y1), (frame_width, Y1), (255, 0, 0), 2)
    cv2.line(frame, (0, Y2), (frame_width, Y2), (0, 0, 255), 2)

    if r.boxes is None or len(r.boxes) == 0:
        cv2.imshow('frame', frame)
        if writer: writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    boxes = r.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    track_id = boxes.id.cpu().numpy() if boxes.id is not None else None

    for i in range(len(xyxy)):
        x1, y1_box, x2, y2_box = xyxy[i].astype(int)
        class_id = int(cls[i])
        class_name = model.names[class_id]
        score = conf[i]
        tid = int(track_id[i]) if track_id is not None else -1

        if tid != -1:
            seen_id_total.add(tid)
            if class_name not in seen_id_class: seen_id_class[class_name] = set()
            seen_id_class[class_name].add(tid)

        label = f'{class_name} {score:.2f}' + (f' Id {tid}' if tid != -1 else '')
        cv2.rectangle(frame, (x1, y1_box), (x2, y2_box), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1_box - th - 10), (x1 + tw + 10, y1_box), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 5, y1_box - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        total = len(seen_id_total)
        cv2.putText(frame, f'unique objects {total}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cx = (x1 + x2) // 2
        cy = (y1_box + y2_box) // 2
        now = time.time()

        if tid != -1:
            if tid not in cross_times:
                cross_times[tid] = {"printed": False, "speed_kmh": None}
            data = cross_times[tid]

            if "dir" not in data:
                if cy < Y1: data["dir"] = 1
                elif cy > Y2: data["dir"] = -1

            if "dir" in data:
                if data["dir"] == 1:
                    if "t1" not in data and cy >= Y1: data["t1"] = now
                    elif "t2" not in data and cy >= Y2: data["t2"] = now
                elif data["dir"] == -1:
                    if "t1" not in data and cy <= Y2: data["t1"] = now
                    elif "t2" not in data and cy <= Y1: data["t2"] = now

                if "t1" in data and "t2" in data and data["speed_kmh"] is None:
                    dt = data["t2"] - data["t1"]
                    if dt != 0:
                        speed_kmh = REAL_DISTANCE_M / abs(dt) * 3.6
                        data["speed_kmh"] = speed_kmh
                        print(f"Vehicle ID {tid} speed: {speed_kmh:.2f} km/h")
                        csv_data.append([tid, class_name, f"{speed_kmh:.2f}"])


            if data["speed_kmh"] is not None:
                cv2.putText(frame, f"{data['speed_kmh']:.2f} km/h", (x1, y1_box - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('frame', frame)
    if writer: writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# Save CSV
with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([len(csv_data)]) 
    writer.writerow(["Vehicle ID", "Class", "Speed (km/h)"])
    writer.writerows(csv_data)

print(f"CSV saved to {CSV_PATH}")
