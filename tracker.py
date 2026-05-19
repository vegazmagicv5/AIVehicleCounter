import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

class VehicleTracker:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # Default line, will be updated dynamically
        self.line_zone = sv.LineZone(start=sv.Point(0, 0), end=sv.Point(1, 0))
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        
        # COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [2, 3, 5, 7]
        self.class_names_dict = self.model.model.names

    def set_line(self, y_pos, width):
        start = sv.Point(0, y_pos)
        end = sv.Point(width, y_pos)
        # Only re-initialize if the line changed to avoid resetting counts
        if (self.line_zone.vector.start.y != y_pos or 
            self.line_zone.vector.end.x != width):
            self.line_zone = sv.LineZone(start=start, end=end)
            
    def set_target_classes(self, class_names):
        # Convert names back to IDs
        name_to_id = {v: k for k, v in self.class_names_dict.items()}
        self.target_classes = [name_to_id[name] for name in class_names if name in name_to_id]

    def process_frame(self, frame):
        # Run YOLOv8 tracking
        results = self.model.track(frame, persist=True, classes=self.target_classes, verbose=False)
        
        if not results or not results[0]:
            return frame, {"in": self.line_zone.in_count, "out": self.line_zone.out_count}
            
        detections = sv.Detections.from_ultralytics(results[0])
        detections = self.tracker.update_with_detections(detections)

        # Trigger line crossing
        self.line_zone.trigger(detections)

        # Annotate
        labels = [
            f"#{tracker_id} {self.class_names_dict[class_id]} {confidence:0.2f}"
            for class_id, tracker_id, confidence
            in zip(detections.class_id, detections.tracker_id, detections.confidence)
        ]
        
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = self.line_annotator.annotate(frame=annotated_frame, line_counter=self.line_zone)

        counts = {
            "in": self.line_zone.in_count,
            "out": self.line_zone.out_count
        }

        return annotated_frame, counts
