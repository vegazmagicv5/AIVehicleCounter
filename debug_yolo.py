import cv2
import supervision as sv
from ultralytics import YOLO

def test():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture("sample_traffic.mp4")
    if not cap.isOpened():
        print("Failed to open video")
        return
        
    frame_count = 0
    total_detections = 0
    
    while cap.isOpened() and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model(frame, classes=[2, 3, 5, 7], verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        
        num_dets = len(detections)
        total_detections += num_dets
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: Found {num_dets} vehicles.")
            
    print(f"Total frames: {frame_count}, Total detections: {total_detections}")

if __name__ == "__main__":
    test()
