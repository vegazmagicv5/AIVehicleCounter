import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

class SimpleTracker:
    """A highly robust distance-based tracker with momentum and memory (buffer)."""
    def __init__(self, max_distance=250, max_age=30):
        self.next_id = 1
        # objects maps id -> (cX, cY, vX, vY, age)
        self.objects = {} 
        self.max_distance = max_distance
        self.max_age = max_age

    def update(self, rects):
        if len(rects) == 0:
            return []

        assigned_ids = [-1] * len(rects)
        new_objects = {}
        
        # 1. Compute all predicted distances
        distances = []
        for i, rect in enumerate(rects):
            cX = (rect[0] + rect[2]) / 2.0
            cY = (rect[1] + rect[3]) / 2.0
            for obj_id, (prev_cX, prev_cY, vX, vY, age) in self.objects.items():
                # Predict next position based on velocity
                pred_cX = prev_cX + vX
                pred_cY = prev_cY + vY
                
                # Distance to PREDICTED position
                dist = ((cX - pred_cX)**2 + (cY - pred_cY)**2)**0.5
                if dist < self.max_distance:
                    distances.append((dist, i, obj_id))
                    
        # 2. Sort by distance (greedy assignment)
        distances.sort(key=lambda x: x[0])
        
        used_rects = set()
        used_objs = set()
        
        for dist, i, obj_id in distances:
            if i in used_rects or obj_id in used_objs:
                continue
                
            rect = rects[i]
            cX = (rect[0] + rect[2]) / 2.0
            cY = (rect[1] + rect[3]) / 2.0
            
            prev_cX, prev_cY, old_vX, old_vY, _ = self.objects[obj_id]
            
            # Update velocity with smoothing (alpha=0.5) to prevent erratic jumps
            curr_vX = cX - prev_cX
            curr_vY = cY - prev_cY
            new_vX = 0.5 * old_vX + 0.5 * curr_vX
            new_vY = 0.5 * old_vY + 0.5 * curr_vY
            
            new_objects[obj_id] = (cX, cY, new_vX, new_vY, 0) # Reset age to 0
            assigned_ids[i] = obj_id
            
            used_rects.add(i)
            used_objs.add(obj_id)
            
        # 3. Assign new IDs to unmatched rects
        for i, rect in enumerate(rects):
            if i not in used_rects:
                cX = (rect[0] + rect[2]) / 2.0
                cY = (rect[1] + rect[3]) / 2.0
                new_objects[self.next_id] = (cX, cY, 0, 0, 0) # Initial velocity and age 0
                assigned_ids[i] = self.next_id
                self.next_id += 1
                
        # 4. Keep old objects that weren't matched, if age < max_age
        for obj_id, (prev_cX, prev_cY, vX, vY, age) in self.objects.items():
            if obj_id not in used_objs:
                if age < self.max_age:
                    # Coast the object forward using momentum even though it's unseen
                    pred_cX = prev_cX + vX
                    pred_cY = prev_cY + vY
                    new_objects[obj_id] = (pred_cX, pred_cY, vX, vY, age + 1)
                
        self.objects = new_objects
        return assigned_ids

class VehicleTracker:
    def __init__(self, model):
        self.model = model
        # Use our custom low-FPS immune tracker instead of ByteTrack
        self.tracker = SimpleTracker(max_distance=300)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # Default line, will be updated dynamically
        self.current_y_pos = -1
        self.current_width = -1
        
        # Manual counting state
        self.track_history = {}
        self.custom_in = 0
        self.custom_out = 0
        
        # COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [2, 3, 5, 7]
        self.class_names_dict = self.model.model.names

    def reset(self):
        if self.current_y_pos != -1:
            pass

    def set_line(self, y_pos, width):
        # Only re-initialize if the line changed to avoid resetting counts
        if self.current_y_pos != y_pos or self.current_width != width:
            self.current_y_pos = y_pos
            self.current_width = width
            
    def set_target_classes(self, class_names):
        # Convert names back to IDs
        name_to_id = {v: k for k, v in self.class_names_dict.items()}
        self.target_classes = [name_to_id[name] for name in class_names if name in name_to_id]

    def process_frame(self, frame):
        # Run YOLOv8 detection with moderate confidence (0.10) 
        # using a higher resolution (imgsz=960) to see Car 4 without making the video lag heavily
        results = self.model(frame, classes=self.target_classes, conf=0.10, imgsz=960, verbose=False)
        
        if not results or not results[0]:
            self._draw_manual_line(frame, 0)
            return frame, {"in": self.custom_in, "out": self.custom_out}
            
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Apply class-agnostic NMS with STRICT threshold (0.5) to completely kill double-counts.
        # Since our tracker now has memory, it's safe if cars briefly disappear when crossing!
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        
        # Guard: skip if no detections
        if len(detections) == 0:
            self._draw_manual_line(frame, 0)
            return frame, {"in": self.custom_in, "out": self.custom_out}
            
        # Apply custom robust tracking
        tracker_ids = self.tracker.update(detections.xyxy)
        detections.tracker_id = np.array(tracker_ids)

        # Bulletproof Manual Counting Logic
        for xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            
            # Guard: skip if tracker_id is None
            if tracker_id is None:
                continue
            
            center_y = float((xyxy[1] + xyxy[3]) / 2.0)
            line_y = float(self.current_y_pos)
            
            if tracker_id not in self.track_history:
                # First time seen — just record position, don't count
                self.track_history[tracker_id] = center_y
            else:
                prev_y = self.track_history[tracker_id]
                
                # Crossed DOWN (In): was above line, now below
                if prev_y < line_y and center_y >= line_y:
                    self.custom_in += 1
                    print(f"[IN] tracker_id={tracker_id} prev_y={prev_y:.1f} -> center_y={center_y:.1f} line={line_y}")
                
                # Crossed UP (Out): was below line, now above
                elif prev_y > line_y and center_y <= line_y:
                    self.custom_out += 1
                    print(f"[OUT] tracker_id={tracker_id} prev_y={prev_y:.1f} -> center_y={center_y:.1f} line={line_y}")
                
                # Always update history
                self.track_history[tracker_id] = center_y

        # Annotate
        labels = [
            f"#{tracker_id} Y:{int((xyxy[1] + xyxy[3]) / 2.0)}"
            for xyxy, tracker_id
            in zip(detections.xyxy, detections.tracker_id)
        ]
        
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        # Draw manual line and counts
        self._draw_manual_line(annotated_frame, len(detections))

        counts = {
            "in": self.custom_in,
            "out": self.custom_out
        }

        return annotated_frame, counts

    def _draw_manual_line(self, frame, current_detections):
        if self.current_y_pos != -1 and self.current_width != -1:
            # Draw line
            cv2.line(frame, (0, self.current_y_pos), (self.current_width, self.current_y_pos), (0, 255, 255), 2)
            # Draw counts with debug info and exact Line Y
            text = f"LINE Y:{self.current_y_pos} | IN:{self.custom_in} OUT:{self.custom_out} | Dets:{current_detections} Tracks:{len(self.track_history)}"
            
            # Add text background for visibility
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (10, self.current_y_pos - 10 - text_height - 5), (10 + text_width, self.current_y_pos - 10 + baseline - 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, self.current_y_pos - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
