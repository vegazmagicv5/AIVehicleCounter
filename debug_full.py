import cv2
import supervision as sv
from ultralytics import YOLO

def test():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture("sample_traffic.mp4")
    if not cap.isOpened():
        print("Failed to open video")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    y_pos = int(height * 0.5)
    print(f"Video {width}x{height}, Line at y={y_pos}")
    
    track_history = {}
    custom_in = 0
    custom_out = 0
    tracker = sv.ByteTrack()
    
    frame_count = 0
    while cap.isOpened() and frame_count < 1000:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model(frame, classes=[2, 3, 5, 7], verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)
        
        for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            center_y = (xyxy[1] + xyxy[3]) / 2.0
            
            if tracker_id not in track_history:
                track_history[tracker_id] = center_y
            else:
                prev_y = track_history[tracker_id]
                
                if prev_y < y_pos and center_y >= y_pos:
                    custom_in += 1
                    print(f"Frame {frame_count}: ID {tracker_id} crossed DOWN (IN) at y={center_y:.1f}")
                elif prev_y > y_pos and center_y <= y_pos:
                    custom_out += 1
                    print(f"Frame {frame_count}: ID {tracker_id} crossed UP (OUT) at y={center_y:.1f}")
                    
                track_history[tracker_id] = center_y
                
    print(f"Total processed: {frame_count} frames. IN={custom_in}, OUT={custom_out}")

if __name__ == "__main__":
    test()
