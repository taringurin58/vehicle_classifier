from ultralytics import YOLO
import cv2
import os
import numpy as np

# โหลดโมเดล YOLO
model = YOLO("yolov8n.pt")

video_path = "../datasets/videos/video4test.mp4"
output_dir = "../datasets/images/train"
os.makedirs(output_dir, exist_ok=True)

# ฟังก์ชันตรวจความเหมือนของภาพ
def is_similar(img1, img2, threshold=15):
    if img1 is None or img2 is None:
        return False
    img1 = cv2.resize(img1, (100, 100))
    img2 = cv2.resize(img2, (100, 100))
    diff = cv2.absdiff(img1, img2)
    mean_diff = np.mean(diff)
    return mean_diff < threshold

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0
last_crop = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # ทุกๆ 10 เฟรมเท่านั้น
    if frame_count % 10 != 0:
        continue

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id in [2, 3, 5, 7] and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            # ตรวจว่าคล้ายกับภาพก่อนหน้าหรือไม่
            if last_crop is not None and is_similar(crop, last_crop):
                continue  # ข้ามถ้าคล้ายกันเกินไป

            save_path = os.path.join(output_dir, f"vehicle_{saved_count:05d}.jpg")
            cv2.imwrite(save_path, crop)
            saved_count += 1
            last_crop = crop

    print(f"Processed frame {frame_count}, total saved: {saved_count}", end="\r")

cap.release()
print(f"\n✅ Done! Saved {saved_count} unique vehicle images to {output_dir}")
