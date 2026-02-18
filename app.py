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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

# ======================== SETTINGS ========================
# Central detection config (shared across machines, tracked in git)
DETECTION_CONFIG_FILE = Path(__file__).parent / "detection_config.json"

def load_detection_config():
    try:
        with open(DETECTION_CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

_det_cfg = load_detection_config()

def _cfg(name, default):
    """Helper to read from env first, then detection_config.json, then default."""
    env_val = os.getenv(name)
    if env_val is not None:
        return env_val
    return str(_det_cfg.get(name.lower(), default))

# Inference runs on EVERY frame (1). Raise to 2-3 if CPU is too slow.
PROCESS_EVERY_N_FRAMES = int(_cfg("PROCESS_EVERY_N_FRAMES", 1))
# Keep full resolution for inference so small/fast people aren't missed.
SCALE_FACTOR = float(_cfg("SCALE_FACTOR", 1.0))
JPEG_QUALITY = int(_cfg("JPEG_QUALITY", 65))                    # JPEG compression quality
# Lower confidence so borderline detections (0.25-0.5) still show up.
DEFAULT_CONFIDENCE = float(_cfg("DEFAULT_CONFIDENCE", 0.25))
LEVEL_2_DURATION_THRESHOLD = float(_cfg("LEVEL_2_DURATION_THRESHOLD", 3.0))

# Optional strict-mode thresholds (per-level minimum confidence)
# Example for very low false positives:
#   LEVEL1_MIN_CONF=0.7
#   LEVEL2_MIN_CONF=0.9
LEVEL1_MIN_CONF = float(_cfg("LEVEL1_MIN_CONF", 0.0))
LEVEL2_MIN_CONF = float(_cfg("LEVEL2_MIN_CONF", 0.0))

# Test-Time Augmentation: runs at multiple scales/flips — greatly improves recall
YOLO_AUGMENT = str(_cfg("AUGMENT", True)).lower() in {"true", "1", "yes"}
# NMS IoU threshold: lower = keep more overlapping boxes (fewer missed people)
YOLO_IOU = float(_cfg("IOU_THRESHOLD", 0.4))

# Performance / hardware overrides
TARGET_FPS = float(_cfg("TARGET_FPS", 30))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto")  # auto | cpu | cuda:0 | 0
# 640 matches what the model was trained at. Lower (320) is faster but less accurate.
YOLO_IMGSZ = int(_cfg("YOLO_IMGSZ", 640))
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

# ======================== DEVICE SETUP ========================
if YOLO_DEVICE.lower() == "auto":
    yolo_device = 0 if torch.cuda.is_available() else "cpu"
else:
    try:
        yolo_device = int(YOLO_DEVICE)
    except ValueError:
        yolo_device = YOLO_DEVICE

use_half = False
if str(YOLO_HALF).lower() in {"true", "1", "yes"}:
    use_half = True
elif str(YOLO_HALF).lower() == "auto":
    use_half = bool(torch.cuda.is_available())

# ======================== ENSEMBLE MODEL LOADING ========================
# Load all model files listed in detection_config.json (or fall back to best.pt)
# Ensemble: run all models on every frame; only fire an alert when
# at least ensemble_min_votes models agree on the same class.

ENSEMBLE_MIN_VOTES = int(_det_cfg.get("ensemble_min_votes", 1))

def _load_models():
    root = Path(__file__).parent
    # Single model override via env var
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        paths = [str(Path(env_path))]
    else:
        configured = _det_cfg.get("model_files", ["best.pt"])
        paths = [str(root / p) for p in configured if (root / p).exists()]
        if not paths:
            paths = [str(root / "best.pt")]

    loaded = []
    for p in paths:
        print(f"🔄 Loading model: {p}")
        try:
            m = YOLO(p)
            try:
                m.fuse()
            except Exception:
                pass
            loaded.append(m)
            print(f"   ✅ Loaded: {Path(p).name}")
        except Exception as e:
            print(f"   ⚠️  Skipped {Path(p).name}: {e}")
    if not loaded:
        raise RuntimeError("No valid model files found. Place at least one .pt file in the project root.")
    print(f"🏁 Ensemble ready: {len(loaded)} model(s), min_votes={ENSEMBLE_MIN_VOTES}")
    return loaded

ensemble_models = _load_models()
# Primary model (first) used for annotated frame rendering
model = ensemble_models[0]

# Display current model metrics
metrics = load_model_metrics()
if metrics["iterations"]:
    _best_m = max(metrics["iterations"], key=lambda x: x.get("f1_score", 0))
    print(f"📊 Best Model: Iteration {_best_m['iteration']} | F1: {_best_m['f1_score']}%")


# Thread pool shared across frames — avoids per-frame thread creation overhead
_ensemble_executor = ThreadPoolExecutor(max_workers=len(ensemble_models) if ensemble_models else 3)

# Cascade mode: run only the primary model on every frame.
# Only call secondary models when the primary already flags a suspicious class.
# This cuts inference cost to ~1 model on calm/empty frames (the majority).
USE_CASCADE = str(_cfg("CASCADE_MODE", True)).lower() in {"true", "1", "yes"}

def _predict_single(m, frame, conf, imgsz, iou, augment, device, half):
    """Run inference on one model — called in a thread."""
    return m.predict(
        frame,
        conf=conf,
        imgsz=imgsz,
        iou=iou,
        augment=augment,
        device=device,
        half=half,
        verbose=False,
    )[0]


def _has_alert(result):
    """Return True if primary model sees ANY person/object at all.

    Intentionally broad: triggers secondary models whenever the primary
    sees anything, so drowning classes that the primary misclassifies as
    Level 0 still get a second opinion from the stronger secondary models.
    Returns False only when the frame is completely empty (no detections).
    """
    return len(result.boxes) > 0


def ensemble_predict(frame, conf, imgsz, iou, augment, device, half):
    """Cascade ensemble predict.

    Step 1 — Run only the primary (fastest) model.
    Step 2 — If it sees nothing suspicious, return immediately (1-model cost).
    Step 3 — Only if primary flags something, run secondary models in parallel
             and apply majority vote.

    Returns (annotated_frame, merged_boxes).
    """
    primary_result = _predict_single(
        ensemble_models[0], frame, conf, imgsz, iou, augment, device, half
    )
    annotated = primary_result.plot()

    # Fast path: nothing suspicious seen by primary — skip secondary models
    if USE_CASCADE and len(ensemble_models) > 1 and not _has_alert(primary_result):
        return annotated, []

    # Slow path: primary flagged something — run secondary models in parallel
    if len(ensemble_models) > 1:
        all_results = [primary_result]
        futures = {
            _ensemble_executor.submit(
                _predict_single, m, frame, conf, imgsz, iou, augment, device, half
            ): i + 1
            for i, m in enumerate(ensemble_models[1:])
        }
        secondary = [None] * len(futures)
        for future in as_completed(futures):
            idx = futures[future] - 1
            try:
                secondary[idx] = future.result()
            except Exception as e:
                print(f"[ENSEMBLE] Secondary model error: {e}")
        all_results += [r for r in secondary if r is not None]
    else:
        all_results = [primary_result]

    # Annotate with the primary model's result
    annotated = all_results[0].plot()

    # Count how many models detected each class
    votes = {}  # class_name -> list of conf scores
    for r in all_results:
        seen_in_this_model = set()
        for box in r.boxes:
            cid = int(box.cls[0])
            try:
                cname = r.names[cid]
            except Exception:
                cname = f"class_{cid}"
            # One vote per class per model (use highest conf for that class)
            if cname not in seen_in_this_model:
                votes.setdefault(cname, []).append(float(box.conf[0]))
                seen_in_this_model.add(cname)

    # Keep only classes that reached the minimum vote threshold
    merged = []
    for cname, confs in votes.items():
        if len(confs) >= ENSEMBLE_MIN_VOTES:
            best_conf = max(confs)
            alert_level, alert_name = get_alert_level_for_class(cname)
            merged.append({
                "class_name": cname,
                "conf": best_conf,
                "votes": len(confs),
                "alert_level": alert_level,
                "alert_name": alert_name,
            })

    return annotated, merged

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
            
            # Run ensemble detection (majority vote across all loaded models)
            annotated_frame, merged_detections = ensemble_predict(
                frame_resized,
                conf=confidence_threshold,
                imgsz=infer_imgsz,
                iou=YOLO_IOU,
                augment=YOLO_AUGMENT,
                device=yolo_device,
                half=use_half,
            )

            # Resize to display size
            if SCALE_FACTOR != 1.0 or display_width != original_width:
                annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))

            # Process detections
            if len(merged_detections) > 0:
                # Debug: print every unique class seen (once per 30 frames to avoid spam)
                if frame_count % 30 == 0:
                    seen = [f"{d['class_name']}({d['conf']:.2f}→L{d['alert_level']} votes={d['votes']})" for d in merged_detections]
                    print(f"[ENSEMBLE frame={frame_count}] {', '.join(seen)}")

                # Pick highest alert level from ensemble-agreed detections
                max_alert_level = 0
                max_conf = 0
                detected_class_name = "Unknown"
                detected_classes = {}

                for d in merged_detections:
                    alert_level = d["alert_level"]
                    conf = d["conf"]
                    class_name = d["class_name"]
                    if alert_level > max_alert_level or (alert_level == max_alert_level and conf > max_conf):
                        max_alert_level = alert_level
                        max_conf = conf
                        detected_class_name = class_name
                    detected_classes[class_name] = {"level": alert_level, "conf": conf, "votes": d["votes"]}
                
                # Process based on highest alert level detected
                if max_alert_level >= 1:  # Level 1 or Level 2
                    conf_percentage = round(max_conf * 100, 2)

                    # Apply strict per-level confidence thresholds (optional)
                    effective_level = max_alert_level
                    if max_alert_level == 2 and LEVEL2_MIN_CONF > 0.0 and max_conf < LEVEL2_MIN_CONF:
                        # Too low for Level 2; downgrade to Level 1 if it meets Level 1 min
                        if LEVEL1_MIN_CONF > 0.0 and max_conf >= LEVEL1_MIN_CONF:
                            effective_level = 1
                        else:
                            # Ignore this detection completely
                            drowning_start_time = None
                            continuous_drowning_frames = 0
                            continue
                    elif max_alert_level == 1 and LEVEL1_MIN_CONF > 0.0 and max_conf < LEVEL1_MIN_CONF:
                        # Ignore weak Level 1 detections
                        drowning_start_time = None
                        continuous_drowning_frames = 0
                        continue

                    # Track duration for Level 2 behaviors (continuous drowning)
                    if effective_level == 2:
                        if drowning_start_time is None:
                            drowning_start_time = current_time
                            continuous_drowning_frames = 1
                        else:
                            continuous_drowning_frames += 1

                        drowning_duration = current_time - drowning_start_time
                    else:
                        # Any frame without Level 2 resets drowning timer
                        drowning_start_time = None
                        continuous_drowning_frames = 0
                        drowning_duration = 0

                    # Set alert type and reason with duration-based Level 2 escalation
                    if effective_level == 2 and drowning_duration >= LEVEL_2_DURATION_THRESHOLD:
                        alert_level = 2
                        alert_type = 'Level 2 - Critical Drowning'
                        reason = f'{detected_class_name} detected for {drowning_duration:.1f}s'
                    elif effective_level == 2:
                        # Treat short Level 2 detections as Level 1 warnings
                        alert_level = 1
                        alert_type = 'Level 1 - Possible Drowning'
                        reason = (
                            f'{detected_class_name} detected for {drowning_duration:.1f}s '
                            f'(< {LEVEL_2_DURATION_THRESHOLD:.1f}s threshold)'
                        )
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
                            'count': len(merged_detections),
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
