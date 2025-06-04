import cv2
from ultralytics import YOLO

# ✅ Set this to True for webcam, False for video file
use_webcam = False

# 📁 If using video file, provide path here
video_path = r"D:\PYTHON 10HRS CWH\PROJECT\newfootball.mp4"

# 🎯 Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 📽️ Initialize video source
cap = cv2.VideoCapture(0 if use_webcam else video_path)

# 🛠️ Video settings (used only for saving video file)
if not use_webcam:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 🎥 VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = r"D:\PYTHON 10HRS CWH\PROJECT\output_detected.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
else:
    out = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of stream or cannot read video.")
        break

    # 🧠 Object detection
    results = model(frame, stream=True)
    people_count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # ✅ Count and color code
            if label == "person":
                people_count += 1
                color = (0, 255, 0)  # Green for people
            else:
                color = (255, 0, 0)  # Blue for others

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 🧮 Display count
    cv2.putText(frame, f"People Count: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 💾 Save frame after detection/annotations
    if out:
        out.write(frame)

    # 🖼️ Show frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # ⛔ Exit with ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
