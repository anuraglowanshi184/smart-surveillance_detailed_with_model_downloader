import cv2
import yaml
from detector import ObjectDetector
from alerts import AlertSystem

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

cap = cv2.VideoCapture(config["camera_source"])
detector = ObjectDetector(config["yolo_model"])
alerter = AlertSystem(method=config["alert_method"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{det['class_id']} Conf:{det['confidence']:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if len(detections) > 0:
        alerter.send_alert("Suspicious activity detected!")

    cv2.imshow("Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
