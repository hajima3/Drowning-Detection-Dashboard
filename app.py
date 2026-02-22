"""
YOLOv11 Drowning Detection Dashboard
Real-time pool safety monitoring with 2-level alert system
"""

from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import time
import os
import json
import re
from pathlib import Path
from collections import deque
import requests
from dotenv import load_dotenv
import torch

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
app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Create required folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/audio', exist_ok=True)

# Load model metrics
METRICS_FILE = Path(__file__).parent / "model_metrics.json"
CLASS_MAPPING_FILE = Path(__file__).parent / "class_mapping.json"

def load_model_metrics():
    """Load deployed model info"""
    try:
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"current_model": "best.pt", "classes": ["Level 0", "Level 1", "Level 2"]}

def load_class_mapping():
    """Load class to alert level mapping"""
    try:
        with open(CLASS_MAPPING_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

# Load class mapping
class_mapping = load_class_mapping()


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

    import datetime
    time_str = datetime.datetime.now().strftime("%H:%M:%S")

    text = (
        "EMERGENCY ALERT\n"
        "Drowning incident detected at Silliman University Pool.\n"
        "Immediate ambulance dispatch required.\n"
        f"Time: {time_str}\n"
        "Location: https://maps.app.goo.gl/1aHKjNd1N1tLyuPM6"
    )

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
    }

    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print(f"[TELEGRAM] Sent Level {level} alert message.")
        else:
            print(f"[TELEGRAM] Failed to send alert. Status: {resp.status_code}, Response: {resp.text}")
    except Exception as e:
        print(f"[TELEGRAM] Error sending alert: {e}")

def     get_alert_level_for_class(class_name):
    """Map a detected class name to an alert level.

    3-level system:
      Level 0 = Swimming  (safe, no action)
      Level 1 = Risky     (monitor, ready to intervene)
      Level 2 = Drowning  (immediate emergency)

    Priority order:
      1. Direct model output labels (fast path — no JSON lookup needed)
      2. class_mapping.json lookup (covers all named sub-classes)
      3. Default to Level 0 if unknown
    """

    def normalize_label(label: str) -> str:
        if not label:
            return ""
        normalized = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    name_key = normalize_label(class_name)

    # ---- Level 2: Drowning (check first — most critical) ----
    if name_key in {
        "drowning", "drown", "drowning_person", "level2_critical",
        "descending_depth", "distress_signs", "erratic_splashing",
        "only_limbs_visible", "physical_collapse",
        "dorwning",   # old model: misspelled
        "lvl2",       # old model: LVL2
        "level_2",    # new model: Level 2
    }:
        return 2, "Drowning"

    # ---- Level 1: Risky ----
    if name_key in {
        "risky", "level1_unsafe", "unsafe", "warning",
        "erratic_unstable_pool_movement",
        "improper_advanced_coordinated_swimming",
        "improper_horizontal_stroke", "improper_movement",
        "improper_swim_wear", "improper_vertical_swimming",
        "unsafe_diving_and_pool_entry",
        "lvl_1",      # old model: LVL 1
        "level_1",    # new model: Level 1
    }:
        return 1, "Risky"

    # ---- Level 0: Swimming ----
    if name_key in {
        "swimming", "swimmer", "normal_swimming", "level0_safe", "safe",
        "back_float", "backstroke", "breaststroke", "butterfly", "dog_paddle",
        "freestyle", "side_stroke", "treading_water", "underwater_swimming",
        "vertical_rest", "entering_pool_behavior",
        "movements_allowed_outside_pool", "proper_swim_wear",
        "lvl_0",      # old model: LVL 0
        "level_0",    # new model: Level 0
    }:
        return 0, "Swimming"

    # ---- Fallback: check class_mapping.json ----
    if class_mapping:
        for level_key, level_data in class_mapping['alert_levels'].items():
            level_classes = level_data.get('classes', [])
            normalized_classes = {normalize_label(c) for c in level_classes}
            if name_key in normalized_classes:
                level_num = int(level_key.split('_')[1])
                return level_num, level_data.get('name', 'Alert')

    # Unknown class — default to Swimming (Level 0)
    return 0, "Swimming"


def _iou(a, b):
    """Compute IoU between two boxes [x1,y1,x2,y2] (display coords)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cross_class_nms(raw_boxes, iou_threshold=0.45):
    """
    Filter overlapping boxes across different classes.
    Keeps the box with the highest alert_level; ties broken by confidence.
    raw_boxes: list of dicts with keys x1,y1,x2,y2,conf,alert_level,class_name
    """
    # Sort: highest alert level first, then highest conf
    sorted_boxes = sorted(raw_boxes, key=lambda d: (d['alert_level'], d['conf']), reverse=True)
    kept = []
    for candidate in sorted_boxes:
        box_c = (candidate['x1'], candidate['y1'], candidate['x2'], candidate['y2'])
        suppressed = False
        for accepted in kept:
            box_a = (accepted['x1'], accepted['y1'], accepted['x2'], accepted['y2'])
            if _iou(box_c, box_a) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return kept


def _smoothed_alert_level(level_window, conf_window):
    """
    Confidence-weighted vote across the last N frames.

    Asymmetric thresholds (safety-first):
      Level 2 fires if it holds >= 25% of weighted votes  (~4 / 15 frames)
      Level 1 fires if it holds >= 35% of weighted votes  (~5 / 15 frames)
      Otherwise Level 0.

    This means:
      - A single noisy frame cannot flip the state
      - Drowning is still caught quickly (4 frames ≈ 0.13 s at 30 fps)
      - False Level 1 chatter is filtered out unless sustained
    """
    if not level_window:
        return 0

    level_weights = {0: 0.0, 1: 0.0, 2: 0.0}
    total_weight = 0.0
    for level, conf in zip(level_window, conf_window):
        # Floor weight at 0.10 so zero-conf (no-detection) frames still count
        w = max(float(conf), 0.10)
        level_weights[level] = level_weights.get(level, 0.0) + w
        total_weight += w

    if total_weight == 0:
        return 0

    l2_frac = level_weights.get(2, 0.0) / total_weight
    l1_frac = level_weights.get(1, 0.0) / total_weight

    # Level 2 needs 35% weighted votes  (~5-6 / 15 frames)
    # Level 1 needs 70% weighted votes  (~10-11 / 15 frames) — very hard to reach by noise
    if l2_frac >= 0.35:
        return 2
    if l1_frac >= 0.70:
        return 1
    return 0

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

# ======================== MODEL LOADING ========================
def _load_model():
    root = Path(__file__).parent
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        model_path = str(Path(env_path))
    else:
        configured = _det_cfg.get("model_files", ["best.pt"])
        names = configured if isinstance(configured, list) else [configured]
        model_path = str(root / "best.pt")
        for name in names:
            candidate = root / name
            if candidate.exists():
                model_path = str(candidate)
                break
    m = YOLO(model_path)
    try:
        m.fuse()
    except Exception:
        pass
    classes = list(m.names.values())
    print(f"🌊  Drowning Detection  |  {Path(model_path).name}  |  {' / '.join(classes)}")
    return m

model = _load_model()


current_source = None
detection_active = False
confidence_threshold = DEFAULT_CONFIDENCE
current_fps = 0.0  # Updated by generate_frames, read by /get_stats

# Detection tracking
latest_detections = []
detection_history = []

# Drowning duration tracking
drowning_start_time = None
continuous_drowning_frames = 0
drowning_miss_frames = 0   # grace-period counter for timer hysteresis

# ======================== VIDEO PROCESSING ========================
def generate_frames(source):
    """Generate frames with YOLOv11 detection"""
    global detection_active, drowning_start_time, continuous_drowning_frames, drowning_miss_frames, current_fps
    
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
    
    print(f"📹 Stream started ({display_width}x{display_height})")
    
    frame_count = 0
    fps_time = time.time()
    last_annotated_frame = None
    last_frame_time = time.time()
    target_fps = TARGET_FPS
    infer_imgsz = YOLO_IMGSZ

    # Temporal smoothing — rolling window of the last 15 processed frames
    SMOOTH_WINDOW = 15
    level_vote_window = deque(maxlen=SMOOTH_WINDOW)  # alert level per frame
    conf_vote_window  = deque(maxlen=SMOOTH_WINDOW)  # max confidence per frame
    drowning_miss_frames = 0  # reset per stream session
    
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
                iou=YOLO_IOU,
                device=yolo_device,
                half=use_half,
                verbose=False,
            )[0]

            # Build base display frame (clean, no YOLO default colors)
            if display_width != original_width or display_height != original_height:
                annotated_frame = cv2.resize(frame, (display_width, display_height))
            else:
                annotated_frame = frame.copy()

            # Scale factors: inference coords → display coords
            scale_x = display_width / process_width
            scale_y = display_height / process_height

            # Process detections
            detections = results.boxes
            if detections is not None and len(detections) > 0:
                max_alert_level = 0
                max_conf = 0
                detected_class_name = "Unknown"

                # --- Step 1: Collect all raw boxes ---
                raw_boxes = []
                for box in detections:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    try:
                        class_name = model.names[class_id]
                    except Exception:
                        class_name = f"class_{class_id}"
                    alert_level, _ = get_alert_level_for_class(class_name)

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw_boxes.append({
                        'x1': int(x1 * scale_x), 'y1': int(y1 * scale_y),
                        'x2': int(x2 * scale_x), 'y2': int(y2 * scale_y),
                        'conf': conf, 'alert_level': alert_level, 'class_name': class_name
                    })

                # --- Step 2: Cross-class NMS (remove duplicate boxes for same region) ---
                filtered_boxes = cross_class_nms(raw_boxes, iou_threshold=0.45)

                # --- Step 3: Draw surviving boxes and track highest alert level ---
                for det in filtered_boxes:
                    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                    conf = det['conf']
                    alert_level = det['alert_level']
                    class_name = det['class_name']

                    # Color per alert level (BGR)
                    if alert_level == 2:
                        box_color = (0, 0, 180)      # Dark Red
                    elif alert_level == 1:
                        box_color = (0, 140, 255)    # Orange
                    else:
                        box_color = (220, 180, 0)    # Blue

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)

                    # Draw label background + text
                    label = f'{class_name} {conf:.2f}'
                    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - bl - 8), (x1 + tw + 6, y1), box_color, -1)
                    cv2.putText(annotated_frame, label, (x1 + 3, y1 - bl - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Track highest alert level
                    if alert_level > max_alert_level or (alert_level == max_alert_level and conf > max_conf):
                        max_alert_level = alert_level
                        max_conf = conf
                        detected_class_name = class_name

                # --- Step 4: Push this frame's result to the smoothing window ---
                level_vote_window.append(max_alert_level)
                conf_vote_window.append(max_conf)

                # Derive consensus alert level across recent frames
                smoothed_level = _smoothed_alert_level(level_vote_window, conf_vote_window)

                # Process based on smoothed (consensus) alert level
                if smoothed_level >= 1:  # Level 1 or Level 2
                    conf_percentage = round(max_conf * 100, 2)

                    # Apply per-level confidence floor (uses raw frame conf as gate)
                    effective_level = smoothed_level
                    if smoothed_level == 2 and LEVEL2_MIN_CONF > 0.0 and max_conf < LEVEL2_MIN_CONF:
                        # Raw conf too low for Level 2; downgrade if Level 1 conf met
                        if LEVEL1_MIN_CONF > 0.0 and max_conf >= LEVEL1_MIN_CONF:
                            effective_level = 1
                        else:
                            # Ignore this detection completely
                            drowning_start_time = None
                            continuous_drowning_frames = 0
                            continue
                    elif smoothed_level == 1 and LEVEL1_MIN_CONF > 0.0 and max_conf < LEVEL1_MIN_CONF:
                        # Ignore weak Level 1 detections
                        drowning_start_time = None
                        continuous_drowning_frames = 0
                        continue

                    # Track duration for Level 2 behaviors (continuous drowning)
                    if effective_level == 2:
                        drowning_miss_frames = 0  # clear grace counter
                        if drowning_start_time is None:
                            drowning_start_time = current_time
                            continuous_drowning_frames = 1
                        else:
                            continuous_drowning_frames += 1

                        drowning_duration = current_time - drowning_start_time
                    else:
                        # Hysteresis: allow up to 20 non-Level-2 frames (~0.67s)
                        # before resetting the drowning countdown.
                        drowning_miss_frames += 1
                        if drowning_miss_frames >= 20:
                            drowning_start_time = None
                            continuous_drowning_frames = 0
                            drowning_miss_frames = 0
                        drowning_duration = (current_time - drowning_start_time) if drowning_start_time else 0

                    # Set alert type and reason with duration-based Level 2 escalation
                    if effective_level == 2 and drowning_duration >= LEVEL_2_DURATION_THRESHOLD:
                        # Full Level 2 emergency — 3+ continuous seconds of drowning
                        alert_level = 2
                        alert_type = 'Level 2 - Drowning'
                        audio_cue = 'alarm'
                        reason = f'{detected_class_name} detected for {drowning_duration:.1f}s'
                        should_fire = True
                    elif effective_level == 2:
                        # Level 2 building up — silent, waiting for 3s threshold
                        # No alert, no beep, no log — just keep the timer running
                        should_fire = False
                        audio_cue = 'none'
                        alert_level = 2
                        alert_type = 'Level 2 - Drowning'
                        reason = ''
                    else:
                        # Genuine Level 1 — beep only, no SMS
                        alert_level = 1
                        alert_type = 'Level 1 - Risky'
                        audio_cue = 'beep'
                        reason = f'{detected_class_name} detected'
                        should_fire = True
                    
                    # Only log and alert if this event should fire
                    if should_fire:
                        # Avoid duplicate events within 2 seconds of same level
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
                                'audio': audio_cue,        # 'alarm' | 'beep' | 'none'
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

                            # Level 2 >= 3s only: trigger Telegram / SMS alert
                            if alert_level == 2:
                                send_telegram_alert(detection_event)
                    
                    # Display alert banner (top-left, clean pill style)
                    if alert_level == 2:
                        bg_color    = (0, 0, 160)      # Dark red
                        accent      = (0, 0, 220)      # Lighter red accent line
                        icon        = '!! DROWNING'
                        detail      = f'{detected_class_name}'
                        if drowning_duration > 0:
                            detail += f'  {drowning_duration:.1f}s'
                    else:
                        bg_color    = (0, 130, 235)    # Orange
                        accent      = (0, 170, 255)    # Lighter orange accent line
                        icon        = 'RISKY'
                        detail      = f'{detected_class_name}  {conf_percentage:.0f}%'

                    pad_x, pad_y = 14, 8
                    font         = cv2.FONT_HERSHEY_SIMPLEX

                    # Measure text sizes
                    (iw, ih), _ = cv2.getTextSize(icon,   font, 0.85, 2)
                    (dw, dh), _ = cv2.getTextSize(detail, font, 0.55, 1)

                    banner_w = max(iw, dw) + pad_x * 2
                    banner_h = ih + dh + pad_y * 3
                    bx, by   = 12, 12

                    # Shadow
                    cv2.rectangle(annotated_frame,
                                  (bx + 3, by + 3),
                                  (bx + banner_w + 3, by + banner_h + 3),
                                  (0, 0, 0), -1)
                    # Main background
                    cv2.rectangle(annotated_frame,
                                  (bx, by),
                                  (bx + banner_w, by + banner_h),
                                  bg_color, -1)
                    # Left accent bar
                    cv2.rectangle(annotated_frame,
                                  (bx, by),
                                  (bx + 5, by + banner_h),
                                  accent, -1)
                    # Icon / title text
                    cv2.putText(annotated_frame, icon,
                                (bx + pad_x, by + pad_y + ih),
                                font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                    # Detail text (slightly smaller, slightly transparent-looking)
                    cv2.putText(annotated_frame, detail,
                                (bx + pad_x, by + pad_y * 2 + ih + dh),
                                font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                else:
                    # Level 0 (smoothed) — count toward grace period
                    drowning_miss_frames += 1
                    if drowning_miss_frames >= 20:
                        drowning_start_time = None
                        continuous_drowning_frames = 0
                        drowning_miss_frames = 0
            else:
                # No detections this frame — vote Level 0, count toward grace period
                level_vote_window.append(0)
                conf_vote_window.append(0.0)
                drowning_miss_frames += 1
                if drowning_miss_frames >= 20:
                    drowning_start_time = None
                    continuous_drowning_frames = 0
                    drowning_miss_frames = 0
            
            # FPS counter — recalculate every 30 frames, draw every frame
            if frame_count % 30 == 0:
                fps_current_time = time.time()
                current_fps = 30 / max(fps_current_time - fps_time, 0.001)
                fps_time = fps_current_time

            # FPS is tracked for /get_stats but not drawn on video
            
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

    # Stop any active stream before switching source
    detection_active = False
    current_source = None

    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400

    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    try:
        filename = secure_filename(file.filename)
        if not filename:
            # Fallback name in case secure_filename strips everything
            filename = f"upload_{int(time.time())}.mp4"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({"status": "error", "message": "File saved but appears empty"}), 500

        drowning_start_time = None
        continuous_drowning_frames = 0
        current_source = filepath
        detection_active = True

        return jsonify({"status": "success", "message": f"Video uploaded: {filename}", "filename": filename})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Upload failed: {str(e)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"status": "error", "message": "File too large (max 500 MB)"}), 413

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
        "fps": round(current_fps, 1),
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
    print(f"🌐  http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
