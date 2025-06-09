from ultralytics import YOLO

person_model = YOLO("yolov8n.pt")

def detect_multiple_people(image_path):
    results = person_model(image_path)[0]
    count = 0
    for r in results.boxes.data.tolist():
        cls_id = int(r[5])
        if results.names[cls_id] == "person":
            count += 1
    return count > 1
