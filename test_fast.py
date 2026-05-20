import cv2
import supervision as sv
from ultralytics import YOLO

def test():
    model = YOLO("yolov8n.pt")
    tracker = sv.ByteTrack()
    
    cap = cv2.VideoCapture("sample_traffic.mp4")
    if not cap.isOpened():
        print("Failed to open video")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    y_pos = int(height * 0.5)
    line_zone = sv.LineZone(start=sv.Point(0, y_pos), end=sv.Point(width, y_pos))
    
    frame_count = 0
    in_count = 0
    out_count = 0
    
    while cap.isOpened() and frame_count < 200:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model(frame, classes=[2, 3, 5, 7], verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)
        
        line_zone.trigger(detections)
        
        if line_zone.in_count > in_count or line_zone.out_count > out_count:
            print(f"Frame {frame_count}: IN={line_zone.in_count}, OUT={line_zone.out_count}", flush=True)
            in_count = line_zone.in_count
            out_count = line_zone.out_count
            
        if frame_count % 10 == 0:
            print(f"Frame {frame_count} IDs: {detections.tracker_id}", flush=True)

if __name__ == "__main__":
    test()
