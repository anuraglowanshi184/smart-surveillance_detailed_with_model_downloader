from flask import Flask, jsonify, render_template, Response, send_file
import datetime, threading, time, os, io, csv
import cv2, numpy as np
from ultralytics import YOLO
import geocoder
from playsound import playsound

app = Flask(__name__)

# -------------------------
# Globals
# -------------------------
alerts_log = []
_next_alert_id = 1
alerts_lock = threading.Lock()
camera_running = False
frame = None
video_writer = None
session_folder = None
all_sessions = []

# -------------------------
# YOLO Model
# -------------------------
model = YOLO("yolov8n.pt")
ALERT_CLASSES = {"knife", "gun", "pistol", "rifle", "firearm", "fire", "flame", "smoke"}
CONF_THRESH = 0.35

# -------------------------
# Helpers
# -------------------------
def add_alert(msg, alert_type="info"):
    """Add new alert with timestamp and type"""
    global _next_alert_id
    with alerts_lock:
        alert = {
            "id": _next_alert_id,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "msg": msg,
            "type": alert_type
        }
        alerts_log.append(alert)
        _next_alert_id += 1
    print(f"Alert: {alert}")
    return alert


def detect_small_fire(frame):
    """Detect small fire regions using HSV color mask"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 150, 150])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
            detected = True
    return detected


def get_gps_location():
    """Fetch approximate GPS coordinates from IP"""
    try:
        g = geocoder.ip('me')
        if g.ok:
            return g.latlng  # [latitude, longitude]
    except Exception as e:
        print("GPS error:", e)
    return [0.0, 0.0]


def save_alerts_csv():
    """Save alerts to CSV inside session folder"""
    global session_folder
    if not alerts_log:
        return
    if not os.path.exists(session_folder):
        os.makedirs(session_folder)
    filename = os.path.join(session_folder, f"alerts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=alerts_log[0].keys())
        writer.writeheader()
        writer.writerows(alerts_log)
    print(f"Alerts saved: {filename}")


# -------------------------
# Camera Detection Thread
# -------------------------
def camera_detection():
    global camera_running, frame, video_writer, session_folder, all_sessions
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        add_alert("Camera failed to open", "info")
        camera_running = False
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20

    session_folder = os.path.join("sessions", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_folder, exist_ok=True)
    all_sessions.append(session_folder)
    video_path = os.path.join(session_folder, "recorded_video.mp4")
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while camera_running:
        ret, f = cap.read()
        if not ret:
            add_alert("Frame read failed", "info")
            break

        results = model(f, verbose=False)
        r = results[0]
        boxes = getattr(r, "boxes", None)

        # --- Object detection ---
        if boxes is not None and len(boxes) > 0:
            names = model.names if hasattr(model, "names") else getattr(r, "names", {})
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for cls_id, conf in zip(cls_ids, confs):
                if conf < CONF_THRESH:
                    continue
                name = str(names.get(cls_id, cls_id)).lower()
                if name in ALERT_CLASSES:
                    alert_type = "fire" if "fire" in name or "flame" in name else "weapon"
                    add_alert(f"{name} detected (conf {conf:.2f})", alert_type)

                    # 🔊 Play beep sound
                    threading.Thread(target=lambda: playsound("beep-beep-43875.mp3"), daemon=True).start()

                    # 📍 Get GPS & draw on frame
                    gps = get_gps_location()
                    if gps != [0.0, 0.0]:
                        lat, lon = gps
                        cv2.rectangle(f, (20, 20), (280, 90), (0, 255, 255), 2)
                        cv2.putText(f, f"GPS: {lat:.4f},{lon:.4f}", (30, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        add_alert(f"GPS Location: {lat:.4f}, {lon:.4f}", "info")

        # --- Small fire color detection ---
        if detect_small_fire(f):
            add_alert("Small flame/matchstick detected", "fire")
            threading.Thread(target=lambda: playsound("beep-beep-43875.mp3"), daemon=True).start()

            gps = get_gps_location()
            if gps != [0.0, 0.0]:
                lat, lon = gps
                cv2.rectangle(f, (20, 20), (280, 90), (0, 255, 255), 2)
                cv2.putText(f, f"GPS: {lat:.4f},{lon:.4f}", (30, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                add_alert(f"GPS Location: {lat:.4f}, {lon:.4f}", "info")

        try:
            frame = r.plot()
        except Exception:
            frame = f.copy()

        if video_writer:
            video_writer.write(frame)
        time.sleep(0.01)

    cap.release()
    if video_writer:
        video_writer.release()
    camera_running = False
    save_alerts_csv()



# Video Streaming

def generate_frames():
    global frame
    while True:
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



# Routes

@app.route('/')
def home():
    return render_template('index.html', sessions=all_sessions)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/alerts', methods=['GET'])
def get_alerts():
    with alerts_lock:
        return jsonify(alerts_log)


@app.route('/start_camera', methods=['GET'])
def start_camera():
    global camera_running
    if camera_running:
        return "Camera already running"
    camera_running = True
    threading.Thread(target=camera_detection, daemon=True).start()
    return "Camera started"


@app.route('/stop_camera', methods=['GET'])
def stop_camera():
    global camera_running
    if not camera_running:
        return "Camera not running"
    camera_running = False
    return "Camera stopping"


@app.route('/download_alerts', methods=['GET'])
def download_alerts():
    with alerts_lock:
        if not alerts_log:
            return "No alerts to download", 404
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=alerts_log[0].keys())
        writer.writeheader()
        writer.writerows(alerts_log)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode('utf-8')),
                         mimetype='text/csv',
                         download_name=f'alerts_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                         as_attachment=True)


@app.route('/download_video', methods=['GET'])
def download_video():
    global session_folder
    if not session_folder:
        return "No video recorded yet", 404
    video_path = os.path.join(session_folder, "recorded_video.mp4")
    if not os.path.exists(video_path):
        return "Video not found", 404
    return send_file(video_path,
                     mimetype='video/mp4',
                     download_name=f'detection_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4',
                     as_attachment=True)


# Main
if __name__ == "__main__":
    add_alert("Smart Surveillance started", "info")
    os.makedirs("sessions", exist_ok=True)
    app.run(debug=True, port=5000)
