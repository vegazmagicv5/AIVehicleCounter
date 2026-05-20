import numpy as np
import supervision as sv

def test():
    tracker = sv.ByteTrack()
    line_zone = sv.LineZone(start=sv.Point(0, 500), end=sv.Point(1920, 500))
    
    # Smooth movement from y=300 to y=700 in steps of 20 pixels
    for y in range(300, 700, 20):
        det = sv.Detections(
            xyxy=np.array([[900, y-50, 1000, y+50]]),
            confidence=np.array([0.9]),
            class_id=np.array([2])
        )
        det = tracker.update_with_detections(det)
        if det.tracker_id is not None and len(det.tracker_id) > 0:
            line_zone.trigger(det)
            print(f"y={y}, id={det.tracker_id[0]}, counts: in={line_zone.in_count}, out={line_zone.out_count}")
        else:
            print(f"y={y}, no ID")

if __name__ == "__main__":
    test()
