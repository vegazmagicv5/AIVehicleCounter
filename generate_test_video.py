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
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    y_pos = int(height * 0.5)
    line_zone = sv.LineZone(start=sv.Point(0, y_pos), end=sv.Point(width, y_pos))
    line_annotator = sv.LineZoneAnnotator()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened() and frame_count < 500:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model(frame, classes=[2, 3, 5, 7], verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)
        
        line_zone.trigger(detections)
        
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = line_annotator.annotate(frame=annotated_frame, line_counter=line_zone)
        
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    print(f"Done. Processed {frame_count} frames. IN={line_zone.in_count}, OUT={line_zone.out_count}")

if __name__ == "__main__":
    test()
