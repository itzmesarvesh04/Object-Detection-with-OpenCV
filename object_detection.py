import cv2
from ultralytics import YOLO

# 📁 Path to the input video file
video_path = r"D:\PYTHON 10HRS CWH\PROJECT\newfootball.mp4"

# 🎯 Load YOLOv8 model
model = YOLO("yolov8n.pt")

# 📽️ Initialize video source
cap = cv2.VideoCapture(video_path)

# 🛠️ Video settings for saving
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#  Set up VideoWriter for saving output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = r"D:\PYTHON 10HRS CWH\PROJECT\output_detected.mp4"
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

#  Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing complete or failed to read frame.")
        break

    #  Run detection
    results = model(frame, stream=True)
    people_count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            #  Count people and draw boxes
            if label == "person":
                people_count += 1
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 🧮 Show people count
    cv2.putText(frame, f"People Count: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 💾 Save and display frame
    out.write(frame)
    cv2.imshow("YOLOv8 Object Detection", frame)

    # ⛔ Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

# 🔚 Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
