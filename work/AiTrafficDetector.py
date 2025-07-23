import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time

CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
CLASSES_OF_INTEREST = [0, 1, 2, 3]

TIME_THRESHOLD = 20
MATCH_THRESHOLD = 10

LINES = [
    ((220, 175), (470, 160)),
    ((200, 175), (235, 260)),
    ((490, 175), (625, 235)),
    ((240, 275), (610, 240))
]
LINE_COLOR = (0, 255, 0)
TRAFFIC_LIGHT_COLORS = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0)
}
TIMER_INTERVAL = 30

class ObjectTracker:
    def __init__(self):
        self.objects = {}
        self.timestamps = {}
        self.crossed = {}
        self.next_id = 0

    def update(self, detections, current_time):
        new_objects = []
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            closest_object = self.match_object(center)
            if closest_object is not None:
                self.objects[closest_object]['bbox'] = detection
                self.objects[closest_object]['center'] = center
                self.timestamps[closest_object] = current_time
                new_objects.append(detection)
            else:
                object_id = self.next_id
                self.objects[object_id] = {'bbox': detection, 'center': center}
                self.timestamps[object_id] = current_time
                self.crossed[object_id] = [False] * len(LINES)  
                self.next_id += 1
                new_objects.append(detection)

        self.cleanup_objects(current_time)
        return new_objects

    def match_object(self, center):
        min_distance = float('inf')
        matched_object = None
        for obj_id, obj in self.objects.items():
            prev_center = obj['center']
            dist = np.linalg.norm(np.array(center) - np.array(prev_center))
            if dist < MATCH_THRESHOLD and dist < min_distance:
                min_distance = dist
                matched_object = obj_id
        return matched_object

    def cleanup_objects(self, current_time):
        objects_to_remove = []
        for obj_id, timestamp in self.timestamps.items():
            if current_time - timestamp > TIME_THRESHOLD:
                objects_to_remove.append(obj_id)

        for obj_id in objects_to_remove:
            del self.objects[obj_id]
            del self.timestamps[obj_id]
            del self.crossed[obj_id]

    def has_crossed(self, object_id, line_index):
        if object_id not in self.crossed:
            return False
        return self.crossed[object_id][line_index]

    def mark_crossed(self, object_id, line_index):
        if object_id not in self.crossed:
            self.crossed[object_id] = [False] * len(LINES)  # Initialize crossed status for new objects
        self.crossed[object_id][line_index] = True


def setup_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path).to(device)
    model.overrides.update({
        'conf': CONFIDENCE_THRESHOLD,
        'iou': IOU_THRESHOLD,
        'classes': CLASSES_OF_INTEREST
    })
    return model


def process_detections(results):
    detections = sv.Detections.from_ultralytics(results)
    mask = np.zeros(len(detections), dtype=bool)
    for i, (confidence, class_id) in enumerate(zip(detections.confidence, detections.class_id)):
        if (class_id in CLASSES_OF_INTEREST and confidence > CONFIDENCE_THRESHOLD):
            mask[i] = True

    filtered_detections = sv.Detections(
        xyxy=detections.xyxy[mask],
        confidence=detections.confidence[mask],
        class_id=detections.class_id[mask]
    )

    return filtered_detections


def is_crossed(line_start, line_end, detection):
    x1, y1, x2, y2 = detection[:4]
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    x_line1, y_line1 = line_start
    x_line2, y_line2 = line_end
    return (min(x_line1, x_line2) < center[0] < max(x_line1, x_line2)) and \
           (min(y_line1, y_line2) < center[1] < max(y_line1, y_line2))


def get_traffic_light_color(vehicle_count):
    if vehicle_count < 5:
        return TRAFFIC_LIGHT_COLORS['green']
    elif 5 <= vehicle_count < 10:
        return TRAFFIC_LIGHT_COLORS['yellow']
    else:
        return TRAFFIC_LIGHT_COLORS['red']


def detect_objects(video_path, model_path):
    model = setup_model(model_path)
    tracker = ObjectTracker()

    annotator = sv.BoxAnnotator(
        thickness=0.5,
        color=sv.Color(255, 0, 0)
    )

    detection_counts = [0] * len(LINES)
    last_cross_times = [0] * len(LINES)
    last_reset_time = time.time()

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        results = model(frame, stream=True, verbose=False)

        for result in results:
            detections = process_detections(result)

            detections_data = []
            for xyxy, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                detections_data.append((*xyxy, class_id, conf))

            tracked_objects = tracker.update(detections_data, current_time)

            for detection in tracked_objects:
                x1, y1, x2, y2 = detection[:4]
                class_id = detection[4]
                conf = detection[5] if len(detection) > 5 else None
                label = f"{model.model.names[class_id]} ({conf:.2f})" if conf is not None else model.model.names[class_id]

                for i, (line_start, line_end) in enumerate(LINES):
                    if is_crossed(line_start, line_end, detection):
                        object_id = len(tracked_objects) - 1
                        if not tracker.has_crossed(object_id, i):
                            detection_counts[i] += 1
                            tracker.mark_crossed(object_id, i)
                            print(f"Vehicle detected crossing Line {i + 1}! Count: {detection_counts[i]}")

                cv2.putText(
                    frame, label, (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            for line_start, line_end in LINES:
                cv2.line(frame, line_start, line_end, LINE_COLOR, 2)

            for i, count in enumerate(detection_counts):
                light_color = get_traffic_light_color(count)
                light_position = (LINES[i][0][0] + LINES[i][1][0]) // 2, (LINES[i][0][1] + LINES[i][1][1]) // 2
                cv2.circle(frame, light_position, 20, light_color, -1)

                time_remaining = TIMER_INTERVAL - int(current_time - last_reset_time)
                cv2.putText(frame, f"{time_remaining}s", (light_position[0] + 30, light_position[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if current_time - last_reset_time > TIMER_INTERVAL:
                detection_counts = [0] * len(LINES)
                last_reset_time = current_time

        cv2.imshow("Traffic Light Simulation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = r"C:\Users\yashv\Music\work\footage2.mp4"
model_path = r"C:\Users\yashv\Music\work\best.pt"

detect_objects(video_path, model_path)
 