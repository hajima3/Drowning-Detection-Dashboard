"""
YOLOv11 Drowning Detection Dashboard
Real-time pool safety monitoring with 2-level alert system
"""

from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import time
import os
import json
import re
from pathlib import Path
import requests
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

# ======================== SETTINGS ========================
# Inference runs on EVERY frame (1). Raise to 2-3 if CPU is too slow.
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "1"))
# Keep full resolution for inference so small/fast people aren't missed.
SCALE_FACTOR = float(os.getenv("SCALE_FACTOR", "1.0"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "65"))                    # JPEG compression quality
# Lower confidence so borderline detections (0.25-0.5) still show up.
DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.25"))
LEVEL_2_DURATION_THRESHOLD = float(os.getenv("LEVEL_2_DURATION_THRESHOLD", "3.0"))

# Performance / hardware overrides
TARGET_FPS = float(os.getenv("TARGET_FPS", "30"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto")  # auto | cpu | cuda:0 | 0
# 640 matches what the model was trained at. Lower (320) is faster but less accurate.
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_HALF = os.getenv("YOLO_HALF", "auto")  # auto | true | false

# ======================== FLASK APP ========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create required folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/audio', exist_ok=True)

# Load model metrics
METRICS_FILE = Path(__file__).parent / "model_metrics.json"
CLASS_MAPPING_FILE = Path(__file__).parent / "class_mapping.json"

def load_model_metrics():
    """Load model iteration metrics"""
    try:
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"current_model": "best.pt", "target_f1": 95.0, "iterations": []}

def load_class_mapping():
    """Load class to alert level mapping"""
    try:
        with open(CLASS_MAPPING_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

# Load class mapping
class_mapping = load_class_mapping()
if class_mapping:
    print("✅ Class mapping loaded successfully!")
    print(f"   Level 0: {len(class_mapping['alert_levels']['level_0']['classes'])} classes")
    print(f"   Level 1: {len(class_mapping['alert_levels']['level_1']['classes'])} classes")
    print(f"   Level 2: {len(class_mapping['alert_levels']['level_2']['classes'])} classes")


def send_telegram_alert(detection_event):
    """Send a Telegram message for Level 2 (critical) alerts, if configured.

    Configuration is taken from environment variables:
    - TELEGRAM_BOT_TOKEN
    - TELEGRAM_CHAT_ID
    """

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("[TELEGRAM] Skipping send: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return

    # Build message text from detection event
    level = detection_event.get("level", 2)
    confidence = detection_event.get("confidence", 0)
    detected_class = detection_event.get("class", "Unknown")
    duration = detection_event.get("duration", 0)

    lines = [
        "🚨 *POOL ALERT: LEVEL 2 DROWNING EMERGENCY*",
        f"*Confidence:* {confidence}%",
        f"*Behavior:* {detected_class}",
    ]
    if duration:
        lines.append(f"*Duration:* {duration}s")

    text = "\n".join(lines)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print(f"[TELEGRAM] Sent Level {level} alert message.")
        else:
            print(f"[TELEGRAM] Failed to send alert. Status: {resp.status_code}, Response: {resp.text}")
    except Exception as e:
        print(f"[TELEGRAM] Error sending alert: {e}")

def get_alert_level_for_class(class_name):
    """Determine alert level based on detected class name.

    Supports both the detailed behavior classes in class_mapping.json and
    simple base classes like "drowning" / "swimming" used in your current model.
    """

    def normalize_label(label: str) -> str:
        if not label:
            return ""
        normalized = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    name_key = normalize_label(class_name)

    # Fallbacks for current 2-class model
    if name_key in {"drowning", "drown", "drowning_person"}:
        return 2, "Critical Drowning"
    if name_key in {"swimming", "swimmer", "normal_swimming"}:
        return 0, "Normal Behavior"

    # If no mapping file loaded, treat as normal
    if not class_mapping:
        return 0, "Normal"

    # Check explicit mappings from class_mapping.json
    for level_key, level_data in class_mapping['alert_levels'].items():
        level_classes = level_data.get('classes', [])
        normalized_classes = {normalize_label(c) for c in level_classes}
        if name_key in normalized_classes:
            level_num = int(level_key.split('_')[1])
            return level_num, level_data.get('name', 'Alert')

    # Default to Level 0 if class not found
    return 0, "Normal"

# Load YOLOv11 model (trained in Label Studio)
# MODEL REQUIREMENTS:
# - Precision: Minimize false positives
# - Recall: Detect all drowning cases  
# - F1 Score: 95-98% target for production
# - Export as 'best.pt' from Label Studio and place in root folder
print("🔄 Loading YOLOv11 model...")
model_path = os.getenv("MODEL_PATH")
if model_path:
    model_path = str(Path(model_path))
else:
    model_path = str(Path(__file__).parent / "best.pt")

if YOLO_DEVICE.lower() == "auto":
    yolo_device = 0 if torch.cuda.is_available() else "cpu"
else:
    # allow either integer device index or explicit string
    try:
        yolo_device = int(YOLO_DEVICE)
    except ValueError:
        yolo_device = YOLO_DEVICE

use_half = False
if str(YOLO_HALF).lower() in {"true", "1", "yes"}:
    use_half = True
elif str(YOLO_HALF).lower() == "auto":
    use_half = bool(torch.cuda.is_available())

model = YOLO(model_path)
try:
    model.fuse()
except Exception:
    pass
print("✅ Model loaded successfully!")

# Display current model metrics
metrics = load_model_metrics()
if metrics["iterations"]:
    best = max(metrics["iterations"], key=lambda x: x.get("f1_score", 0))
    print(f"📊 Best Model: Iteration {best['iteration']} | F1: {best['f1_score']}%")

# ======================== GLOBAL VARIABLES ========================
current_source = None
detection_active = False
confidence_threshold = DEFAULT_CONFIDENCE

# Detection tracking
latest_detections = []
detection_history = []

# Drowning duration tracking
drowning_start_time = None
continuous_drowning_frames = 0

# ======================== VIDEO PROCESSING ========================
def generate_frames(source):
    """Generate frames with YOLOv11 detection"""
    global detection_active, drowning_start_time, continuous_drowning_frames
    
    # Initialize video capture
    if source == 0:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # DirectShow for webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(source)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce lag
    
    # Get video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    process_width = int(original_width * SCALE_FACTOR)
    process_height = int(original_height * SCALE_FACTOR)
    
    max_display_width = 960
    max_display_height = 540
    
    if original_width > max_display_width or original_height > max_display_height:
        display_scale = min(max_display_width / original_width, max_display_height / original_height)
        display_width = int(original_width * display_scale)
        display_height = int(original_height * display_scale)
    else:
        display_width = original_width
        display_height = original_height
    
    print(f"📹 Original: {original_width}x{original_height}")
    print(f"🔄 Processing: {process_width}x{process_height}")
    print(f"📺 Display: {display_width}x{display_height}")
    
    frame_count = 0
    fps_time = time.time()
    last_annotated_frame = None
    last_frame_time = time.time()
    target_fps = TARGET_FPS
    infer_imgsz = YOLO_IMGSZ
    
    while detection_active:
        success, frame = cap.read()
        if not success:
            if source != 0:  # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        # Drop buffered frames for webcam
        if source == 0:
            for _ in range(2):
                cap.grab()
        
        frame_count += 1
        
        # Frame rate limiting
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < (1.0 / target_fps):
            time.sleep((1.0 / target_fps) - elapsed)
        last_frame_time = time.time()
        
        # Skip frames or reuse last detection
        # On skipped frames we still annotate the CURRENT raw frame with the last
        # known boxes so the display doesn't go stale or show nothing.
        if frame_count % PROCESS_EVERY_N_FRAMES != 0 and last_annotated_frame is not None:
            # Re-draw last result boxes onto fresh current frame for smooth video
            annotated_frame = last_annotated_frame
        else:
            # Resize for faster processing
            if SCALE_FACTOR != 1.0:
                frame_resized = cv2.resize(frame, (process_width, process_height))
            else:
                frame_resized = frame
            
            # Run YOLOv11 detection
            results = model.predict(
                frame_resized,
                conf=confidence_threshold,
                imgsz=infer_imgsz,
                device=yolo_device,
                half=use_half,
                verbose=False,
            )[0]
            annotated_frame = results.plot()
            
            # Resize to display size
            if SCALE_FACTOR != 1.0 or display_width != original_width:
                annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))
            
            # Process detections
            detections = results.boxes
            if len(detections) > 0:
                # Debug: print every unique class seen (once per 30 frames to avoid spam)
                if frame_count % 30 == 0:
                    seen = []
                    for _b in detections:
                        _cid = int(_b.cls[0])
                        _cn = model.names.get(_cid, f"class_{_cid}")
                        _lv, _ = get_alert_level_for_class(_cn)
                        seen.append(f"{_cn}({float(_b.conf[0]):.2f}→L{_lv})")
                    print(f"[DETECT frame={frame_count}] {', '.join(seen)}")

                # Track all detected classes and their alert levels
                detected_classes = {}
                max_alert_level = 0
                max_conf = 0
                detected_class_name = "Unknown"
                
                for box in detections:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name from model
                    try:
                        class_name = model.names[class_id]
                    except Exception:
                        class_name = f"class_{class_id}"
                    
                    # Determine alert level for this class
                    alert_level, alert_name = get_alert_level_for_class(class_name)
                    
                    # Track highest alert level
                    if alert_level > max_alert_level or (alert_level == max_alert_level and conf > max_conf):
                        max_alert_level = alert_level
                        max_conf = conf
                        detected_class_name = class_name
                    
                    if class_name not in detected_classes:
                        detected_classes[class_name] = {'level': alert_level, 'conf': conf, 'count': 1}
                    else:
                        detected_classes[class_name]['count'] += 1
                        detected_classes[class_name]['conf'] = max(detected_classes[class_name]['conf'], conf)
                
                # Process based on highest alert level detected
                if max_alert_level >= 1:  # Level 1 or Level 2
                    conf_percentage = round(max_conf * 100, 2)
                    
                    # Track duration for Level 2 behaviors
                    if max_alert_level == 2:
                        if drowning_start_time is None:
                            drowning_start_time = current_time
                            continuous_drowning_frames = 1
                        else:
                            continuous_drowning_frames += 1
                        
                        drowning_duration = current_time - drowning_start_time
                    else:
                        drowning_duration = 0
                    
                    # Set alert type and reason
                    if max_alert_level == 2:
                        alert_level = 2
                        alert_type = 'Level 2 - Critical Drowning'
                        reason = f'{detected_class_name} detected'
                    else:
                        alert_level = 1
                        alert_type = 'Level 1 - Unsafe Movement'
                        reason = f'{detected_class_name} detected'
                    
                    # Log detection (avoid duplicates within 2 seconds)
                    should_log = True
                    if detection_history:
                        last_detection = detection_history[-1]
                        time_diff = current_time - last_detection['timestamp']
                        if time_diff < 2.0 and last_detection['level'] == alert_level:
                            should_log = False
                    
                    if should_log:
                        detection_event = {
                            'type': alert_type,
                            'level': alert_level,
                            'confidence': conf_percentage,
                            'timestamp': current_time,
                            'class': detected_class_name,
                            'count': len(detections),
                            'duration': round(drowning_duration, 1) if drowning_duration > 0 else 0,
                            'reason': reason
                        }
                        detection_history.append(detection_event)
                        latest_detections.append(detection_event)
                        
                        # Keep only last 50 detections
                        if len(detection_history) > 50:
                            detection_history.pop(0)

                        # For Level 2 alerts, trigger Telegram notification (if configured)
                        if alert_level == 2:
                            send_telegram_alert(detection_event)
                    
                    # Display alert
                    text_scale = display_width / original_width
                    
                    if alert_level == 2:
                        color = (0, 0, 255)  # Red
                        label = f'🚨 LEVEL 2 EMERGENCY: {detected_class_name}'
                        if drowning_duration > 0:
                            label += f' | {drowning_duration:.1f}s'
                    else:
                        color = (0, 165, 255)  # Orange
                        label = f'⚠ LEVEL 1 WARNING: {detected_class_name}'
                    
                    rect_width = int(650 * text_scale)
                    rect_height = int(65 * text_scale)
                    cv2.rectangle(annotated_frame, (5, 5), (rect_width, rect_height), color, -1)
                    cv2.putText(annotated_frame, label, (10, int(45 * text_scale)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7 * text_scale, (255, 255, 255), 
                               max(2, int(2 * text_scale)))
                else:
                    # Level 0 (Normal) - reset tracking
                    drowning_start_time = None
                    continuous_drowning_frames = 0
            else:
                # No detections - reset tracking
                drowning_start_time = None
                continuous_drowning_frames = 0
            
            # FPS counter
            if frame_count % 30 == 0:
                fps_current_time = time.time()
                fps = 30 / (fps_current_time - fps_time)
                fps_time = fps_current_time
                
                fps_x = display_width - int(160 * (display_width / original_width))
                cv2.rectangle(annotated_frame, (fps_x, 5), (display_width-5, 45), (50, 50, 50), -1)
                cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (fps_x+10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            last_annotated_frame = annotated_frame
        
        # Encode frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    print("✅ Stream ended")

# ======================== ROUTES ========================
@app.route('/')
def index():
    """Serve dashboard"""
    return render_template('dashboard_live.html')

@app.route('/video_feed')
def video_feed():
    """Stream video with detection"""
    global current_source
    if current_source is not None:
        return Response(generate_frames(current_source),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response("No source selected", status=400)

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start webcam detection"""
    global current_source, detection_active, drowning_start_time, continuous_drowning_frames
    
    cap_test = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_test.set(cv2.CAP_PROP_FPS, 30)
    cap_test.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_test.release()
    
    drowning_start_time = None
    continuous_drowning_frames = 0
    
    current_source = 0
    detection_active = True
    return jsonify({"status": "success", "message": "Webcam started"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global detection_active, current_source, drowning_start_time, continuous_drowning_frames
    detection_active = False
    current_source = None
    drowning_start_time = None
    continuous_drowning_frames = 0
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload and process video"""
    global current_source, detection_active, drowning_start_time, continuous_drowning_frames
    
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        drowning_start_time = None
        continuous_drowning_frames = 0
        
        current_source = filepath
        detection_active = True
        
        return jsonify({"status": "success", "message": f"Video uploaded: {file.filename}"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Upload failed: {str(e)}"}), 500

@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    """Update confidence threshold"""
    global confidence_threshold
    data = request.get_json()
    conf = data.get('confidence', 50)
    confidence_threshold = float(conf) / 100
    return jsonify({"status": "success", "confidence": confidence_threshold * 100})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get detection statistics"""
    return jsonify({
        "active": detection_active,
        "confidence": confidence_threshold * 100,
        "source": "Webcam" if current_source == 0 else ("Video" if current_source else "None")
    })

@app.route('/get_detections', methods=['GET'])
def get_detections():
    """Get latest detections for logging"""
    global latest_detections
    
    detections = latest_detections.copy()
    latest_detections.clear()
    
    return jsonify({
        "status": "success",
        "detections": detections
    })

@app.route('/get_model_metrics', methods=['GET'])
def get_model_metrics():
    """Get model iteration metrics"""
    metrics = load_model_metrics()
    return jsonify(metrics)

@app.route('/get_class_mapping', methods=['GET'])
def get_class_mapping():
    """Get class to alert level mapping"""
    mapping = load_class_mapping()
    if mapping:
        return jsonify(mapping)
    return jsonify({"error": "Class mapping not found"}), 404

# ======================== MAIN ========================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌊 YOLOv11 Drowning Detection Dashboard")
    print("=" * 60)
    print("📡 Starting server...")
    print("🌐 Open: http://localhost:5000")
    print("✅ Ready!\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
