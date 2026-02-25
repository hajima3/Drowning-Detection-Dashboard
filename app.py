"""
YOLOv11 Drowning Detection Dashboard
Real-time pool safety monitoring with 3-level alert system

28-class model consolidated to 3 output levels:
  Level 0 - Swimming  (14 classes): safe, no action
  Level 1 - Risky     (12 classes): monitor, audio beep, 5s escalation timer
  Level 2 - Drowning  ( 2 classes): immediate emergency, alarm + Telegram
    Fires via: (a) direct model detection (Level2_Critical / Physical Collapse, conf >= 80%)
               (b) temporal escalation   (Level 1 sustained >= 5 seconds, conf >= 80%)

10/20 Rule applied:
  - Level 1 must be sustained 5s before escalating (identification window)
  - 20-second response countdown shown on screen once Level 2 fires
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
# Lower confidence so borderline detections (0.20-0.5) still show up.
# 0.20 maximises recall so the model catches more swimmers before applying alert logic.
DEFAULT_CONFIDENCE = float(_cfg("DEFAULT_CONFIDENCE", 0.20))
# 10/20 rule: Level 1 must be sustained for 5s before escalating to Level 2
LEVEL_2_DURATION_THRESHOLD = float(_cfg("LEVEL_2_DURATION_THRESHOLD", 5.0))
# 10/20 rule: lifeguard has 20s to reach the swimmer once Level 2 fires
RESPONSE_COUNTDOWN = float(_cfg("RESPONSE_COUNTDOWN", 20.0))
# Number of consecutive missed-detection frames before boxes clear and timers reset.
# 60 frames = 2 seconds at 30 fps — keeps boxes on screen during brief model misses.
GRACE_PERIOD_FRAMES = int(_cfg("GRACE_PERIOD_FRAMES", 60))

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

def get_alert_level_for_class(class_name):
    """Map a detected class name to an alert level.

    28-CLASS MODEL — 3 OUTPUT LEVELS:
      Level 0 = Swimming  (safe, no action)            - 14 classes
      Level 1 = Risky     (monitor, ready to intervene) - 12 classes
      Level 2 = Drowning  (immediate emergency)         - 2 classes (direct model output)

    Level 2 fires in two ways:
      a) DIRECT: model detects Level2_Critical or Physical Collapse → instant alert
      b) TEMPORAL: Level 1 sustained for 2 seconds continuously → escalated alert

    Priority order:
      1. Level 2 direct classes (highest — never downgraded)
      2. Level 1 risky classes
      3. Level 0 safe classes
      4. class_mapping.json lookup (fallback)
      5. Default to Level 0 if unknown
    """

    def normalize_label(label: str) -> str:
        if not label:
            return ""
        normalized = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    name_key = normalize_label(class_name)

    # ---- Level 2: Drowning / Critical (DIRECT MODEL OUTPUT — highest priority) ----
    if name_key in {
        "level2_critical", "physical_collapse",
        "drowning", "critical",           # generic fallback variants
        "lvl_2", "lvl2", "level_2",
    }:
        return 2, "Drowning"

    # ---- Level 1: Risky (MODEL OUTPUT) ----
    if name_key in {
        "risky", "level1_unsafe", "unsafe", "warning",
        "descending_depth", "distress_signs",
        "erratic_unstable_pool_movement", "erratic_splashing",
        "improper_advanced_coordinated_swimming",
        "improper_horizontal_stroke", "improper_movement",
        "improper_swim_wear", "improper_vertical_swimming",
        "only_limbs_visible", "unsafe_diving_and_pool_entry",
        "lvl_1", "lvl1", "level_1",
    }:
        return 1, "Risky"

    # ---- Level 0: Swimming (MODEL OUTPUT) ----
    if name_key in {
        "swimming", "swimmer", "normal_swimming", "level0_safe", "safe",
        "back_float", "backstroke", "breaststroke", "butterfly", "dog_paddle",
        "freestyle", "side_stroke", "treading_water", "underwater_swimming",
        "vertical_rest", "entering_pool_behavior",
        "movements_allowed_outside_pool", "proper_swim_wear",
        "lvl_0", "lvl0", "level_0",
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
      Level 2 fires if it holds >= 35% of weighted votes  (~5-6 / 15 frames)
      Level 1 fires if it holds >= 40% of weighted votes  (~6 / 15 frames)
      Otherwise Level 0.

    Rationale (tuned to best.pt's 87.7% recall, 82% precision, 28 classes):
      - Level 1 at 40%: responsive to real risky behavior; 12 Level 1 classes
        produce varied labels frame-to-frame so a lower threshold is needed
        to accumulate enough votes while staying robust against single noisy frames.
      - Level 2 at 35%: kept lower because the 80% confidence gate and
        level2_triggered flag already control false Level 2 alerts.
      - A single noisy frame (1/15 = 6.7%) still cannot flip either state.
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
    # Level 1 needs 40% weighted votes  (~6 / 15 frames) — tuned for 12-class Level 1 pool
    if l2_frac >= 0.35:
        return 2
    if l1_frac >= 0.40:
        return 1
    return 0

# ======================== BOX DRAWING HELPER ========================
# Human-readable display names per alert level shown in bounding box labels
_LEVEL_DISPLAY = {0: 'Safe', 1: 'Risky', 2: 'Drowning'}

def _draw_detection_boxes(frame, boxes):
    """Draw bounding boxes + labels onto frame in-place. Works for any list of
    box dicts produced by cross_class_nms (keys: x1,y1,x2,y2,conf,alert_level,class_name).

    Label format:  [L1] Risky  0.57
    The level display name (Safe / Risky / Drowning) is always derived from
    alert_level so a downgraded detection never shows a misleading class name.
    """
    for det in boxes:
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        conf        = det['conf']
        alert_level = det['alert_level']

        # Color and level tag per effective alert level (BGR)
        if alert_level == 2:
            box_color = (0, 0, 220)      # Bright Red   — Level 2 emergency
            level_tag = '[L2]'
        elif alert_level == 1:
            box_color = (0, 140, 255)    # Orange       — Level 1 risky
            level_tag = '[L1]'
        else:
            box_color = (220, 180, 0)    # Blue         — Level 0 safe
            level_tag = '[L0]'

        # Bounding box (thicker border for Level 2)
        thickness = 4 if alert_level == 2 else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Label uses the level display name — never the raw model class name —
        # so a downgraded Level2_Critical (conf<80%) reads "[L1] Risky  0.57"
        level_name = _LEVEL_DISPLAY.get(alert_level, 'Unknown')
        label = f'{level_tag} {level_name}  {conf:.2f}'
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - bl - 8), (x1 + tw + 6, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - bl - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


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
level2_triggered = False   # Flag to ensure only ONE Level 2 log & SMS per incident
level2_trigger_time = None # Timestamp when Level 2 first fired (for 20s response countdown)

# ======================== VIDEO PROCESSING ========================
def generate_frames(source):
    """Generate frames with YOLOv11 detection"""
    global detection_active, drowning_start_time, continuous_drowning_frames, drowning_miss_frames, current_fps, level2_triggered, level2_trigger_time
    
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

    # Persist last known boxes so they are redrawn on every frame (including
    # skipped inference frames) — keeps boxes visible whenever a person is present.
    last_known_boxes = []

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
        
        # On skipped inference frames: build a fresh display frame and overlay
        # the last known boxes — this keeps boxes visible and correctly positioned
        # on the current video frame rather than showing a stale frozen frame.
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            if display_width != original_width or display_height != original_height:
                annotated_frame = cv2.resize(frame, (display_width, display_height))
            else:
                annotated_frame = frame.copy()
            # Always draw last known boxes so they stay on screen
            if last_known_boxes:
                _draw_detection_boxes(annotated_frame, last_known_boxes)
            last_annotated_frame = annotated_frame
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

                    # Confidence downgrade: Level 2 below 80% → treat as Level 1
                    # (affects box color AND smoothing vote)
                    if alert_level == 2 and conf < 0.80:
                        alert_level = 1

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw_boxes.append({
                        'x1': int(x1 * scale_x), 'y1': int(y1 * scale_y),
                        'x2': int(x2 * scale_x), 'y2': int(y2 * scale_y),
                        'conf': conf, 'alert_level': alert_level, 'class_name': class_name
                    })

                # --- Step 2: Cross-class NMS (remove duplicate boxes for same region) ---
                filtered_boxes = cross_class_nms(raw_boxes, iou_threshold=YOLO_IOU)

                # Persist boxes so skipped frames can redraw them
                last_known_boxes = filtered_boxes

                # --- Step 3: Draw surviving boxes and track highest alert level ---
                _draw_detection_boxes(annotated_frame, filtered_boxes)

                for det in filtered_boxes:
                    alert_level = det['alert_level']
                    conf        = det['conf']
                    class_name  = det['class_name']

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
                    if smoothed_level == 1 and LEVEL1_MIN_CONF > 0.0 and max_conf < LEVEL1_MIN_CONF:
                        # Conf below floor: keep timers running but suppress alert this frame.
                        # Do NOT reset drowning timer — sustained low-conf Level 1 is still risky.
                        # Do NOT skip frame encode — that stutters the video stream.
                        effective_level = 0  # treat as no-alert for this frame only

                    # ========== ALERT LEVEL LOGIC ==========
                    # Level 2 fires in two ways:
                    #   a) DIRECT: model detects Level2_Critical / Physical Collapse
                    #   b) TEMPORAL: Level 1 sustained for 2s → escalated
                    # Level 1: risky behavior — audio beep, start duration timer
                    # Level 0: safe — reset timer, no alert
                    #
                    # CONFIDENCE DOWNGRADE: if effective_level is 2 but confidence
                    # is below 80%, treat it as Level 1 (risky) instead.
                    if effective_level == 2 and max_conf < 0.80:
                        effective_level = 1  # downgrade — not confident enough for critical

                    if effective_level == 2:
                        # ---- DIRECT Level 2 from model (Level2_Critical / Physical Collapse) ----
                        drowning_miss_frames = 0
                        drowning_duration = (current_time - drowning_start_time) if drowning_start_time else 0
                        alert_level = 2
                        alert_type = 'Level 2 - Drowning'
                        audio_cue = 'alarm'
                        reason = f'{detected_class_name} - CRITICAL DETECTED DIRECTLY BY MODEL'
                        # Fire once per incident (confidence already >= 80% from gate above)
                        should_fire = not level2_triggered

                    elif effective_level == 1:
                        # Level 1 detected — start or continue tracking duration
                        drowning_miss_frames = 0  # clear grace counter
                        if drowning_start_time is None:
                            drowning_start_time = current_time
                            continuous_drowning_frames = 1
                        else:
                            continuous_drowning_frames += 1

                        drowning_duration = current_time - drowning_start_time

                        # TEMPORAL ESCALATION: Level 1 sustained for 5+ seconds → Level 2
                        if drowning_duration >= LEVEL_2_DURATION_THRESHOLD:
                            # Escalate to Level 2 (Drowning) - EMERGENCY!
                            alert_level = 2
                            alert_type = 'Level 2 - Drowning'
                            audio_cue = 'alarm'
                            reason = f'Risky behavior sustained for {drowning_duration:.1f}s - ESCALATED TO DROWNING'
                            # Fire once per incident.
                            # NOTE: No 80% conf gate here — the 5s sustained duration IS
                            # the quality gate for temporal escalation. The 80% gate only
                            # applies to direct Level2_Critical / Physical Collapse detections.
                            should_fire = not level2_triggered
                        else:
                            # Level 1 building up - show but don't escalate yet
                            alert_level = 1
                            alert_type = 'Level 1 - Risky'
                            audio_cue = 'beep'
                            reason = f'{detected_class_name} detected ({drowning_duration:.1f}s)'
                            should_fire = True
                    else:
                        # Level 0 (Swimming) - hysteresis grace period before reset
                        # Allow up to GRACE_PERIOD_FRAMES (~2s) before resetting timer
                        drowning_miss_frames += 1
                        if drowning_miss_frames >= GRACE_PERIOD_FRAMES:
                            drowning_start_time = None
                            continuous_drowning_frames = 0
                            drowning_miss_frames = 0
                            level2_triggered = False   # Reset — incident over
                            level2_trigger_time = None # Clear countdown
                        drowning_duration = (current_time - drowning_start_time) if drowning_start_time else 0

                        # Don't fire alert for Level 0 (safe swimming)
                        should_fire = False
                    
                    # Only log and alert if this event should fire
                    if should_fire:
                        # Level 2: one log per incident guaranteed by level2_triggered flag +
                        #          80% confidence gate already applied in should_fire above.
                        # Level 1: avoid duplicate events within 2 seconds.
                        should_log = True
                        if alert_level == 1 and detection_history:
                            last_detection = detection_history[-1]
                            time_diff = current_time - last_detection['timestamp']
                            if time_diff < 2.0 and last_detection['level'] == 1:
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

                            # Level 2: trigger Telegram / SMS alert & set flags
                            if alert_level == 2:
                                send_telegram_alert(detection_event)
                                level2_triggered = True          # One log per incident
                                level2_trigger_time = current_time  # Start 20s response countdown
                    
                    # Display alert banner (top-left, clean pill style)
                    if alert_level == 2:
                        # LEVEL 2: DROWNING EMERGENCY (direct model detection OR temporal escalation)
                        bg_color    = (0, 0, 160)      # Dark red
                        accent      = (0, 0, 220)      # Lighter red accent line
                        icon        = '!! DROWNING EMERGENCY'
                        # 10/20 rule: show 20-second response countdown
                        if level2_trigger_time is not None:
                            response_remaining = max(0.0, RESPONSE_COUNTDOWN - (current_time - level2_trigger_time))
                            if effective_level == 2:
                                detail = f'{detected_class_name}  {conf_percentage:.0f}% - RESPOND IN {response_remaining:.0f}s'
                            else:
                                detail = f'Risky sustained {drowning_duration:.0f}s - RESPOND IN {response_remaining:.0f}s'
                        else:
                            detail  = f'{detected_class_name}  {conf_percentage:.0f}%'
                    elif alert_level == 1:
                        # LEVEL 1: RISKY (show detection timer: Xs / 10s to escalation)
                        bg_color    = (0, 130, 235)    # Orange
                        accent      = (0, 170, 255)    # Lighter orange accent line
                        icon        = 'RISKY BEHAVIOR'
                        if drowning_duration > 0:
                            detail = f'{detected_class_name}  {conf_percentage:.0f}% - {drowning_duration:.1f}s/{LEVEL_2_DURATION_THRESHOLD:.0f}s'
                        else:
                            detail = f'{detected_class_name}  {conf_percentage:.0f}%'
                    else:
                        # LEVEL 0: SAFE (shouldn't reach here but handle gracefully)
                        bg_color    = (220, 180, 0)    # Blue (safe)
                        accent      = (255, 200, 0)    # Lighter blue
                        icon        = 'SAFE'
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
                    if drowning_miss_frames >= GRACE_PERIOD_FRAMES:
                        drowning_start_time = None
                        continuous_drowning_frames = 0
                        drowning_miss_frames = 0
                        level2_triggered = False   # Reset — incident over
                        level2_trigger_time = None # Clear countdown
            else:
                # No detections this frame — vote Level 0, count toward grace period
                level_vote_window.append(0)
                conf_vote_window.append(0.0)
                drowning_miss_frames += 1
                if drowning_miss_frames >= GRACE_PERIOD_FRAMES:
                    drowning_start_time = None
                    continuous_drowning_frames = 0
                    drowning_miss_frames = 0
                    level2_triggered = False   # Reset — incident over
                    level2_trigger_time = None # Clear countdown
                    last_known_boxes = []      # Clear boxes — person has gone
                elif last_known_boxes:
                    # Within grace period: keep drawing last known boxes on the
                    # fresh frame so the box doesn't flicker when detection briefly drops
                    _draw_detection_boxes(annotated_frame, last_known_boxes)
            
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
    global current_source, detection_active, drowning_start_time, continuous_drowning_frames, level2_triggered, level2_trigger_time
    
    cap_test = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_test.set(cv2.CAP_PROP_FPS, 30)
    cap_test.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_test.release()
    
    drowning_start_time = None
    continuous_drowning_frames = 0
    level2_triggered = False
    level2_trigger_time = None
    
    current_source = 0
    detection_active = True
    return jsonify({"status": "success", "message": "Webcam started"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global detection_active, current_source, drowning_start_time, continuous_drowning_frames, level2_triggered, level2_trigger_time
    detection_active = False
    current_source = None
    drowning_start_time = None
    continuous_drowning_frames = 0
    level2_triggered = False
    level2_trigger_time = None
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload and process video"""
    global current_source, detection_active, drowning_start_time, continuous_drowning_frames, level2_triggered, level2_trigger_time

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
        level2_triggered = False
        level2_trigger_time = None
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
