"""
YOLOv11 Drowning Detection Dashboard — Live Testing Build
  Level 0: swimming / floating  — safe, no action
  Level 1: drowning detected    — orange box + beep + escalation timer starts
  Level 2: sustained drowning   — red box + alarm + Telegram + response countdown
"""

import logging
import datetime
import contextlib
import threading
import json

logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

from flask import Flask, render_template, Response, request, jsonify
import flask.cli
flask.cli.show_server_banner = lambda *args, **kwargs: None

from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import time
import os
import math
from collections import deque
from pathlib import Path
import requests
from dotenv import load_dotenv
import torch
import numpy as np

load_dotenv()
cv2.setNumThreads(8)   # i7-12700: 8 P-cores for video decode/resize

# ======================== LOAD CONFIG FILES ========================
_BASE = Path(__file__).parent

with open(_BASE / 'detection_config.json', encoding='utf-8-sig') as _f:
    _cfg = json.load(_f)
_inf    = _cfg['inference']
_alert  = _cfg['alert']
_stream = _cfg['stream']

with open(_BASE / 'class_mapping.json', encoding='utf-8-sig') as _f:
    _cmap = json.load(_f)
# Maps class name → alert_level. Used for both level assignment and NMS priority
# (higher alert_level = higher priority: drowning > floating > swimming).
_CLASS_LEVEL: dict[str, int] = {
    v['name'].lower(): v['alert_level']
    for v in _cmap['classes'].values()
}

with open(_BASE / 'model_metrics.json', encoding='utf-8-sig') as _f:
    _metrics = json.load(_f)

# ── Class vote weights from model metrics ────────────────────────────────────
# drowning_weight = recall/precision ≈ 1.006  (compensates for recall > precision)
# safe_weight     = 1 - FNR           ≈ 0.988
_m_precision = _metrics['training']['precision']   # 0.981
_m_recall    = _metrics['training']['recall']       # 0.988

_CLASS_VOTE_WEIGHT: dict[str, float] = {
    cls: (_m_recall / _m_precision) if _CLASS_LEVEL.get(cls, 0) >= 1 else _m_recall
    for cls in _CLASS_LEVEL
}

# Monotonic counter assigned to every new track — drift-proof ID matching.
_next_track_id: int = 0

# ======================== SETTINGS ========================
# detection_config.json is the source of truth; .env overrides for live tuning.
INFERENCE_CONF      = float(os.getenv('DEFAULT_CONFIDENCE',        _inf['confidence']))
YOLO_IOU            = float(os.getenv('IOU_THRESHOLD',             _inf['iou']))
YOLO_IMGSZ          = int(os.getenv('YOLO_IMGSZ',                  _inf['imgsz']))

# Alert thresholds
DROWNING_CONF_MIN   = float(os.getenv('LEVEL1_MIN_CONF',           _alert['drowning_conf_min']))    # conf floor to call a box L1 (0.40)

# Temporal smoothing + consecutive-frame threshold
CONSEC_MIN_FRAMES       = int(os.getenv('CONSEC_MIN_FRAMES',        6))    # consecutive present+future drowning frames to confirm (~0.4s)
MIN_DOMINANCE           = float(os.getenv('MIN_DOMINANCE',          0.55)) # drowning must hold ≥55% of total W@M score to trigger alert

# Escalation state machine
CONFIRM_SECONDS  = float(os.getenv('CONFIRM_SECONDS',             _alert.get('confirm_seconds', 2.0)))
ESCALATION_TIME  = float(os.getenv('LEVEL_2_DURATION_THRESHOLD', _alert['escalation_time_s']))
RESPONSE_COUNTDOWN = float(os.getenv('RESPONSE_COUNTDOWN',       _alert['response_countdown_s']))
GRACE_TIME       = float(os.getenv('GRACE_TIME',                 _alert['grace_time_s']))

# Anti-flicker hysteresis
CONFIRM_DROPOUT_S    = float(os.getenv('CONFIRM_DROPOUT_S',  1.0))   # L1 must drop this long to reset confirmation
EXIT_DOMINANCE_FRAC  = 0.60   # once L1, exit only if dominance < MIN_DOMINANCE × this
EXIT_CONSEC_MIN      = 2      # once L1, need consecutive drowning < this to exit

# Box tracker
TRACK_IOU_MIN  = float(os.getenv('TRACK_IOU_MIN',  0.45))   # min IoU to link detection to existing track
DECAY_RATE     = float(os.getenv('DECAY_RATE',     0.80))   # time-decay: w = e^(-DECAY_RATE × age_s)
MIN_L1_S       = float(os.getenv('MIN_L1_S',       0.50))   # min track age (s) before L1 is allowed
GHOST_TIME_S   = float(os.getenv('GHOST_TIME_S',   0.80))   # drop track after this many seconds unseen
POS_EMA_ALPHA  = float(os.getenv('POS_EMA_ALPHA',  0.70))   # EMA weight for new box position

# Prediction buffer: Past=10 / Present=30 / Future=60 frames @ 30 FPS
# Past    = 0.33 s oldest context        weight 0.50
# Present = 1.00 s = display delay lag   weight 1.00
# Future  = 2.00 s real-time lookahead   weight 1.50
# Score = W @ M  (weight vector × observation matrix per track).
FUTURE_FRAMES  = int(os.getenv('FUTURE_FRAMES',   60))
PRESENT_FRAMES = int(os.getenv('PRESENT_FRAMES',  30))
PAST_FRAMES    = int(os.getenv('PAST_FRAMES',     10))
BEHAV_WINDOW_S = float(os.getenv('BEHAV_WINDOW_S', 4.00))   # must cover full 100-frame buffer
FUTURE_WEIGHT  = float(os.getenv('FUTURE_WEIGHT',  1.50))
PRESENT_WEIGHT = float(os.getenv('PRESENT_WEIGHT', 1.00))
PAST_WEIGHT    = float(os.getenv('PAST_WEIGHT',    0.50))

# Stream / display
DISPLAY_DELAY_S = float(os.getenv('DISPLAY_DELAY_S', 1.00))  # display lag = 1 s = 30 frames

TARGET_FPS          = min(float(os.getenv('TARGET_FPS', _stream['target_fps'])), 30.0)  # hard cap at 30 FPS
INFER_EVERY         = int(os.getenv('INFER_EVERY',                 _stream['infer_every']))
JPEG_QUALITY        = int(os.getenv('JPEG_QUALITY',                _stream['jpeg_quality']))
DISP_MAX_W          = int(_stream['max_display_width'])
DISP_MAX_H          = int(_stream['max_display_height'])

# ======================== FLASK APP ========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(str(_BASE / 'static' / 'audio'), exist_ok=True)

# ======================== TELEGRAM ========================
def send_telegram_alert():
    """Fire-and-forget — spawns a background thread so the stream is never blocked."""
    threading.Thread(target=_do_send_telegram, daemon=True).start()

def _do_send_telegram():
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id   = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("[TELEGRAM] Credentials not set - skipping.")
        return
    time_str = datetime.datetime.now().strftime("%H:%M:%S")
    text = (
        "EMERGENCY ALERT\n"
        "Drowning incident detected at Silliman Pool!\n"
        f"Time: {time_str}\n"
        "Location: https://maps.app.goo.gl/1aHKjNd1N1tLyuPM6"
    )
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=5,
        )
        print(f"[TELEGRAM] {'Sent' if resp.status_code == 200 else 'Failed'}: {resp.status_code}")
    except Exception as e:
        print(f"[TELEGRAM] Error: {e}")

# ======================== CLASS -> LEVEL ========================
def class_to_level(class_name: str, conf: float) -> int:
    """
    Map a detection to an alert level using class_mapping.json.
    Drowning at conf >= DROWNING_CONF_MIN → Level 1.
    Everything else (swimming, floating, low-conf drowning) → Level 0 safe.
    """
    base_level = _CLASS_LEVEL.get(class_name.strip().lower(), 0)
    if base_level == 1 and conf < DROWNING_CONF_MIN:
        return 0   # confident gate: below threshold falls back to L0
    return base_level

# ======================== BOX HELPERS ========================
def _iou(a, b):
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

def _update_tracks(tracks: list, detections: list, now: float = 0.0) -> list:
    """
    Time-based behavioral tracker with exponential time-decay scoring.

    Each track keeps BEHAV_WINDOW_S (4s) of (timestamp, cls, conf) history.
    Class resolved via: weight = e^(-DECAY_RATE × age_s) × vote_weight × conf.
    Min-time gate: track must exist MIN_L1_S before L1 is allowed.

    Note: this tracker drives position smoothing and adaptive inference rate.
    The authoritative display label comes from _pred_buf_resolve (raw YOLO history).
    """
    used = set()

    for t in tracks:
        best_iou, best_i = 0.0, -1
        tb = (t['x1'], t['y1'], t['x2'], t['y2'])
        for i, d in enumerate(detections):
            if i in used:
                continue
            iou = _iou(tb, (d['x1'], d['y1'], d['x2'], d['y2']))
            if iou > best_iou:
                best_iou, best_i = iou, i

        if best_iou >= TRACK_IOU_MIN:
            d = detections[best_i]
            used.add(best_i)
            t['last_seen'] = now

            # Smooth box position
            a = POS_EMA_ALPHA
            t['x1'] = int(a * d['x1'] + (1 - a) * t['x1'])
            t['y1'] = int(a * d['y1'] + (1 - a) * t['y1'])
            t['x2'] = int(a * d['x2'] + (1 - a) * t['x2'])
            t['y2'] = int(a * d['y2'] + (1 - a) * t['y2'])

            # Append timestamped behavioural observation
            t['history'].append((now, d['cls'], d['conf']))

        # Prune observations older than BEHAV_WINDOW_S
        cutoff = now - BEHAV_WINDOW_S
        t['history'] = [(ts, c, cf) for (ts, c, cf) in t['history'] if ts >= cutoff]

        if not t['history']:
            # No evidence at all — reset to safe
            t['cls'] = list(_CLASS_LEVEL.keys())[0]
            t['conf'] = 0.0
            t['alert_level'] = 0
            continue

        # Time-decay scoring: w = e^(-DECAY_RATE × age) × vote_weight × conf
        cls_scores:  dict = {}
        cls_weights: dict = {}
        for (ts, cls, conf) in t['history']:
            age = max(0.0, now - ts)
            w   = math.exp(-DECAY_RATE * age) * _CLASS_VOTE_WEIGHT.get(cls, 1.0) * conf
            cls_scores[cls]  = cls_scores.get(cls,  0.0) + conf * w
            cls_weights[cls] = cls_weights.get(cls, 0.0) + w

        resolved_cls  = max(cls_scores, key=cls_scores.__getitem__)
        resolved_conf = cls_scores[resolved_cls] / cls_weights[resolved_cls]

        t['cls']         = resolved_cls
        t['conf']        = resolved_conf
        t['alert_level'] = class_to_level(resolved_cls, resolved_conf)

        # --- Min-time gate ---
        # Track must have behavioural observations spanning at least MIN_L1_S seconds.
        # A person appearing for 0.1s cannot trigger L1 regardless of confidence.
        time_span = t['history'][-1][0] - t['history'][0][0] if len(t['history']) > 1 else 0.0
        if time_span < MIN_L1_S:
            t['alert_level'] = 0

    # Promote unmatched detections to new tracks
    global _next_track_id
    for i, d in enumerate(detections):
        if i not in used:
            tracks.append({
                'id':          _next_track_id,
                'x1': d['x1'], 'y1': d['y1'], 'x2': d['x2'], 'y2': d['y2'],
                'conf': d['conf'],
                'alert_level': 0,        # always start at L0
                'cls': d['cls'],
                'history': [(now, d['cls'], d['conf'])],
                'last_seen': now,
            })
            _next_track_id += 1

    # Drop tracks not seen within GHOST_TIME_S seconds
    return [t for t in tracks if (now - t.get('last_seen', now)) <= GHOST_TIME_S]

def _merge_tracks(tracks: list) -> list:
    """Merge overlapping tracks (IoU >= 0.50). Keep higher alert_level, combine histories."""
    skip = set()
    for i, a in enumerate(tracks):
        if i in skip:
            continue
        ba = (a['x1'], a['y1'], a['x2'], a['y2'])
        for j, b in enumerate(tracks):
            if j <= i or j in skip:
                continue
            bb = (b['x1'], b['y1'], b['x2'], b['y2'])
            if _iou(ba, bb) >= 0.50:
                dominant = a if (a['alert_level'], a['conf']) >= (b['alert_level'], b['conf']) else b
                absorb   = b if dominant is a else a
                dominant['history'].extend(absorb['history'])
                dominant['history'].sort(key=lambda e: e[0])
                skip.add(j if dominant is a else i)
    return [t for i, t in enumerate(tracks) if i not in skip]

def _pred_buf_resolve(pred_buffer, snap_data: list, last_resolved: dict = None,
                      last_resolved_lvl: dict = None) -> list:
    """
    Matrix-scored class resolution with 5-input AND gate.

    Per track, builds observation matrix  M[frame, class]  of raw confidences.
    Temporal weight vector  W[frame]  = Past(0.5) / Present(1.0) / Future(1.5).
    Class scores = W @ M  (matrix dot product per track).

    Drowning triggers L1 only when ALL 5 AND-gate inputs are HIGH:
      G1) argmax       — drowning has the highest W@M score
      G2) conf_floor   — mean drowning confidence ≥ DROWNING_CONF_MIN
      G3) dominance    — drowning holds ≥ MIN_DOMINANCE of total score
      G4) consecutive  — ≥ CONSEC_MIN_FRAMES consecutive drowning in present+future
      G5) dual_region  — drowning detected in BOTH present AND future windows

    If ANY gate is LOW (False), alert_level stays 0 regardless of score.
    Tie-break: score → mean_conf → future_count → retain previous class.
    """
    if not snap_data or not pred_buffer:
        return []
    if last_resolved is None:
        last_resolved = {}

    sorted_preds = sorted(pred_buffer, key=lambda e: e[0])
    n = len(sorted_preds)

    past_count    = min(PAST_FRAMES, n)
    present_count = min(PRESENT_FRAMES, max(0, n - past_count))

    # Consistent class ordering for matrix columns
    _cls_names = sorted(_CLASS_LEVEL.keys())
    _cls_idx   = {c: i for i, c in enumerate(_cls_names)}
    n_cls      = len(_cls_names)

    # Build frame weight vector  W[frame]  and region flags
    W = np.empty(n, dtype=np.float32)
    _is_present = np.zeros(n, dtype=bool)
    _is_future  = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < past_count:
            W[i] = PAST_WEIGHT
        elif i < past_count + present_count:
            W[i] = PRESENT_WEIGHT
            _is_present[i] = True
        else:
            W[i] = FUTURE_WEIGHT
            _is_future[i] = True

    # Pre-compute max consecutive drowning run per track (present+future only)
    _consec: dict = {}
    _run:    dict = {}
    for i in range(past_count, n):
        _, preds = sorted_preds[i]
        seen_drown = {t for (t, c, _) in preds if _CLASS_LEVEL.get(c, 0) >= 1}
        for t in set(_run.keys()) | seen_drown:
            if t in seen_drown:
                _run[t] = _run.get(t, 0) + 1
                if _run[t] > _consec.get(t, 0):
                    _consec[t] = _run[t]
            else:
                _run[t] = 0

    # Collect all detections per track in a single pass
    snap_tids = {tid for (tid, *_) in snap_data}
    track_obs: dict = {}   # tid → [(frame_idx, cls_idx, conf), ...]
    for f_idx, (_, preds) in enumerate(sorted_preds):
        for (tid, cls, conf) in preds:
            if tid in snap_tids and cls in _cls_idx:
                track_obs.setdefault(tid, []).append((f_idx, _cls_idx[cls], conf))

    # Resolve displayed class per snapshot track
    result = []
    for (tid, sx1, sy1, sx2, sy2) in snap_data:
        obs = track_obs.get(tid)
        if not obs:
            continue

        # Build observation matrix  M[frame, class]  for this track
        M = np.zeros((n, n_cls), dtype=np.float32)
        raw_confs:    dict = {}   # cls_name → [conf, ...]
        present_cnt:  dict = {}   # cls_name → count in present region
        future_cnt:   dict = {}   # cls_name → count in future region

        for (f_idx, ci, conf) in obs:
            M[f_idx, ci] = conf
            cn = _cls_names[ci]
            raw_confs.setdefault(cn, []).append(conf)
            if _is_present[f_idx]:
                present_cnt[cn] = present_cnt.get(cn, 0) + 1
            if _is_future[f_idx]:
                future_cnt[cn] = future_cnt.get(cn, 0) + 1

        # Matrix calculation:  scores = W @ M → (n_cls,)
        scores = W @ M
        total  = float(scores.sum())
        if total < 1e-6:
            continue

        # argmax with tie-breaking
        resolved_cls = max(_cls_names, key=lambda c: (
            float(scores[_cls_idx[c]]),                                       # weighted score
            (sum(raw_confs[c]) / len(raw_confs[c])) if raw_confs.get(c)       # mean raw conf
                else 0.0,
            future_cnt.get(c, 0),                                            # future evidence
            1 if last_resolved.get(tid) == c else 0,                         # stickiness
        ))

        r_confs = raw_confs.get(resolved_cls, [])
        resolved_conf = (sum(r_confs) / len(r_confs)) if r_confs else 0.01
        resolved_conf = max(0.01, min(resolved_conf, 1.0))

        base_level = _CLASS_LEVEL.get(resolved_cls.strip().lower(), 0)

        # ── 5-INPUT AND GATE ──────────────────────────────────────
        # All 5 must be True for drowning (L1) to pass.
        # If ANY is False, output is L0 (safe).
        if base_level >= 1:
            # G1: argmax — drowning already won class selection above
            g2_conf_floor = resolved_conf >= DROWNING_CONF_MIN                # mean conf ≥ DROWNING_CONF_MIN
            dominance     = float(scores[_cls_idx[resolved_cls]]) / total
            g3_dominance  = dominance >= MIN_DOMINANCE                        # ≥ MIN_DOMINANCE of total score
            g4_consec     = _consec.get(tid, 0) >= CONSEC_MIN_FRAMES          # ≥ CONSEC_MIN_FRAMES consecutive frames
            g5_dual       = (present_cnt.get(resolved_cls, 0) >= 1            # in present region
                             and future_cnt.get(resolved_cls, 0) >= 1)        # AND in future region

            all_gates = (g2_conf_floor and g3_dominance and g4_consec and g5_dual)
            prev_lvl = (last_resolved_lvl or {}).get(tid, 0)

            if all_gates:
                alert_level = 1
            elif prev_lvl >= 1:
                # Hysteresis: once L1, use relaxed exit thresholds to prevent flicker
                still_dominant = dominance >= (MIN_DOMINANCE * EXIT_DOMINANCE_FRAC)
                still_consec   = _consec.get(tid, 0) >= EXIT_CONSEC_MIN
                alert_level = 1 if (still_dominant and still_consec) else 0
            else:
                alert_level = 0
        else:
            alert_level = 0

        if last_resolved_lvl is not None:
            last_resolved_lvl[tid] = alert_level
        last_resolved[tid] = resolved_cls
        result.append({
            'x1': sx1, 'y1': sy1, 'x2': sx2, 'y2': sy2,
            'alert_level': alert_level, 'conf': resolved_conf,
        })

    return result

def cross_class_nms(boxes, iou_threshold=0.50):
    """Suppress overlapping boxes across classes. Higher alert_level wins."""
    boxes = sorted(
        boxes,
        key=lambda d: (d['alert_level'], _CLASS_LEVEL.get(d['cls'].lower(), 0), d['conf']),
        reverse=True,
    )
    kept = []
    for candidate in boxes:
        bc = (candidate['x1'], candidate['y1'], candidate['x2'], candidate['y2'])
        if not any(_iou(bc, (k['x1'], k['y1'], k['x2'], k['y2'])) >= iou_threshold for k in kept):
            kept.append(candidate)
    return kept

def draw_boxes(frame, boxes, frame_alert_level=0, incident_box=None):
    """Draw detection boxes. During L2, only the incident person shows red."""
    for d in boxes:
        lvl = d['alert_level']

        # Determine display level for this box
        if frame_alert_level == 2 and incident_box is not None:
            is_incident = _iou(
                (d['x1'], d['y1'], d['x2'], d['y2']), incident_box
            ) >= 0.15
            display_lvl = 2 if is_incident else lvl
        else:
            display_lvl = lvl

        # L1 below DROWNING_CONF_MIN is shown as L0 (not yet confirmed drowning).
        # Once a visual alert is active (frame_alert_level >= 1), always show L1/L2
        # even if conf momentarily dips below the gate.
        if display_lvl == 1 and d['conf'] < DROWNING_CONF_MIN and frame_alert_level == 0:
            display_lvl = 0

        if display_lvl == 2:
            color = (0, 0, 220)          # red — L2 EMERGENCY
            label = f"[L2] Drowning {d['conf']:.2f}"
        elif display_lvl == 1:
            color = (0, 140, 255)        # orange — L1 Risky (>= 40% conf)
            label = f"[L1] Risky {d['conf']:.2f}"
        else:
            color = (255, 130, 0)        # blue — L0 Safe (swimming, floating, or low-conf drowning)
            label = f"[L0] Safe {d['conf']:.2f}"

        x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - bl - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - bl - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

def draw_banner(frame, alert_level, conf_pct, drowning_duration, level2_trigger_time):
    """Draw the top-left status banner. No banner at Level 0."""
    if alert_level == 2:
        bg     = (0, 0, 160);    accent = (0, 0, 220)
        icon   = '!! DROWNING EMERGENCY  |  10/20 STANDARD'
        remaining = max(0.0, RESPONSE_COUNTDOWN - (time.time() - (level2_trigger_time or time.time())))
        detail = (
            f'Sustained {drowning_duration:.0f}s  |  '
            f'20-SEC RESPONSE WINDOW  |  '
            f'REACH VICTIM IN {remaining:.0f}s'
        )
    elif alert_level == 1:
        bg     = (0, 130, 235);  accent = (0, 170, 255)
        icon   = f'DROWNING DETECTED  |  {ESCALATION_TIME:.0f}-SEC CONFIRMATION WINDOW'
        detail = (
            f'{conf_pct:.0f}% confidence  |  '
            f'escalating in {max(0.0, ESCALATION_TIME - drowning_duration):.1f}s'
        )
    else:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    (iw, ih), _ = cv2.getTextSize(icon,   font, 0.85, 2)
    (dw, dh), _ = cv2.getTextSize(detail, font, 0.55, 1)
    px, py = 14, 8
    bw = max(iw, dw) + px * 2
    bh = ih + dh + py * 3
    bx, by = 12, 12
    cv2.rectangle(frame, (bx + 3, by + 3), (bx + bw + 3, by + bh + 3), (0, 0, 0), -1)  # shadow
    cv2.rectangle(frame, (bx, by),         (bx + bw, by + bh),           bg,        -1)  # background
    cv2.rectangle(frame, (bx, by),         (bx + 5,  by + bh),           accent,    -1)  # accent bar
    cv2.putText(frame, icon,   (bx + px, by + py + ih),           font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (bx + px, by + py * 2 + ih + dh),  font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

# ======================== DEVICE + MODEL ========================
_dev = os.getenv("YOLO_DEVICE", "auto")
if _dev.lower() == "auto":
    yolo_device = 0 if torch.cuda.is_available() else "cpu"
else:
    try:    yolo_device = int(_dev)
    except: yolo_device = _dev

model = YOLO(str(Path(__file__).parent / "best.pt"), verbose=False)
with open(os.devnull, 'w') as _dn, contextlib.redirect_stdout(_dn):
    try: model.fuse()
    except: pass

# _use_half: respect detection_config.json 'half' setting AND require GPU
_use_half = _inf.get('half', True) and yolo_device != 'cpu'
torch.backends.cudnn.benchmark = True   # RTX 3060: cuDNN auto-tunes kernels for fixed 640 input
_dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
for _ in range(3):   # 3 warmup frames to fully settle cuDNN kernel selection
    model.predict(_dummy, conf=INFERENCE_CONF, imgsz=YOLO_IMGSZ, iou=YOLO_IOU,
                  device=yolo_device, half=_use_half, verbose=False)
del _dummy

# ======================== GLOBAL STATE ========================
current_source      = None
detection_active    = False
current_fps         = 0.0
current_alert_level = 0   # exposed via /get_stats so the frontend can stop audio on reset

latest_detections = []   # consumed by /get_detections
detection_history = []   # last 50 events (debounced)

# Escalation state — all reset by _reset_state() on stream start/stop
confirm_start_time   = None   # when continuous L1 began
drowning_start_time  = None   # when escalation timer started (post-confirmation)
last_drowning_time   = None   # most recent L1 display frame time
level2_triggered     = False  # prevents duplicate Telegram sends
level2_trigger_time  = None   # when L2 first fired (for countdown)
incident_box         = None   # (x1,y1,x2,y2) of the person who triggered L2

def _reset_state():
    """Clear all escalation state — called on stream start/stop."""
    global confirm_start_time, drowning_start_time, last_drowning_time
    global level2_triggered, level2_trigger_time, incident_box
    confirm_start_time   = None
    drowning_start_time  = None
    last_drowning_time   = None
    level2_triggered     = False
    level2_trigger_time  = None
    incident_box         = None

# ======================== DETECTION LOG ========================
def _log_event(level, audio, conf_pct):
    """Append detection event; debounce Level 1 to once per 2s."""
    now = time.time()
    if level == 1 and detection_history:
        last = detection_history[-1]
        if last['level'] == 1 and (now - last['timestamp']) < 2.0:
            return
    event = {
        'type':       'Level 2 - EMERGENCY' if level == 2 else 'Level 1 - Drowning Detected',
        'level':      level,
        'audio':      audio,
        'confidence': round(conf_pct, 2),
        'timestamp':  now,
    }
    detection_history.append(event)
    latest_detections.append(event)
    if len(detection_history) > 50:
        detection_history.pop(0)

# ======================== VIDEO STREAM ========================
def generate_frames(source):
    global detection_active
    global current_fps, current_alert_level
    global confirm_start_time, drowning_start_time, last_drowning_time
    global level2_triggered, level2_trigger_time, incident_box

    # ---- Capture setup ----
    if source == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG: faster USB bandwidth than YUY2
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISP_MAX_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISP_MAX_H)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale display to max 1280x720 (RTX 3060 streams this cleanly)
    scale  = min(DISP_MAX_W / orig_w, DISP_MAX_H / orig_h, 1.0)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)

    print(f"Stream started ({disp_w}x{disp_h})")

    frame_count     = 0
    fps_time        = time.time()
    last_frame_time = time.time()

    last_filtered   = []    # cached YOLO result for skipped frames
    last_raw_lvl    = 0     # raw YOLO alert level (drives adaptive inference rate)
    report_conf_ema = 0.0   # EMA-smoothed conf for banner/API
    tracks          = []    # box tracker

    raw_frame_buffer: deque = deque()     # (timestamp, pixel_frame, snap_data)
    pred_buffer: deque = deque(maxlen=PAST_FRAMES + PRESENT_FRAMES + FUTURE_FRAMES)
    last_resolved_cls = {}  # tid → last displayed class (stickiness tie-breaker)
    last_resolved_lvl = {}  # tid → last alert level (hysteresis anti-flicker)

    while detection_active:
        ok, frame = cap.read()
        if not ok:
            if source != 0:     # loop video file
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_count += 1
        now = time.time()

        # ---- Throttle to TARGET_FPS ----
        wait = (1.0 / TARGET_FPS) - (now - last_frame_time)
        if wait > 0:
            time.sleep(wait)
        last_frame_time = now = time.time()

        # ---- Resize for display (never draw on the raw buffer) ----
        disp = cv2.resize(frame, (disp_w, disp_h)) if scale != 1.0 else frame.copy()

        # ---- Adaptive inference rate ----
        # Run YOLO every frame when L1/L2 is active so the tracker gets fresh data
        # as fast as possible. Drop back to INFER_EVERY when everything is L0.
        is_hot = last_raw_lvl >= 1 or drowning_start_time is not None
        current_infer_every = 1 if is_hot else INFER_EVERY

        # ---- YOLO inference ----
        if frame_count % current_infer_every == 0:
            results = model.predict(
                disp,
                conf=INFERENCE_CONF,
                imgsz=YOLO_IMGSZ,
                iou=YOLO_IOU,
                device=yolo_device,
                half=_use_half,
                verbose=False,
            )[0]

            raw_boxes = []
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id   = int(box.cls[0])
                    conf     = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    lvl      = class_to_level(cls_name, conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw_boxes.append({
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                        'conf': conf, 'alert_level': lvl, 'cls': cls_name,
                    })

            last_filtered = cross_class_nms(raw_boxes)
            tracks        = _update_tracks(tracks, last_filtered, now=now)
            tracks        = _merge_tracks(tracks)
            last_raw_lvl  = max((b['alert_level'] for b in last_filtered), default=0)

            # Push raw YOLO class/conf (from history, not gated t['cls']) to pred_buffer
            pred_entry = [
                (t.get('id', idx),
                 t['history'][-1][1] if t['history'] else t['cls'],
                 t['history'][-1][2] if t['history'] else t['conf'])
                for idx, t in enumerate(tracks)
            ]
            pred_buffer.append((now, pred_entry))

        # Freeze box positions + track IDs for display buffer (class resolved at pop time)
        snap_data = [(t.get('id', idx), t['x1'], t['y1'], t['x2'], t['y2'])
                     for idx, t in enumerate(tracks)]
        raw_frame_buffer.append((now, disp.copy(), snap_data))

        # ---- FPS counter ----
        if frame_count % 30 == 0:
            t = time.time()
            current_fps = 30 / max(t - fps_time, 0.001)
            fps_time = t

        # ---- Delayed display + state machine ----
        if raw_frame_buffer and (now - raw_frame_buffer[0][0]) >= DISPLAY_DELAY_S:
            pop_ts, display_frame, snap_data = raw_frame_buffer.popleft()
            disp_now = time.time()

            d_drawlist = _pred_buf_resolve(pred_buffer, snap_data, last_resolved_cls, last_resolved_lvl)

            d_max_lvl  = max((d['alert_level'] for d in d_drawlist), default=0)
            d_max_conf = max(
                (d['conf'] for d in d_drawlist if d['alert_level'] == d_max_lvl),
                default=0.0,
            )
            d_incident = None
            if d_max_lvl == 1:
                inc = max((d for d in d_drawlist if d['alert_level'] == 1),
                          key=lambda d: d['conf'], default=None)
                if inc:
                    d_incident = (inc['x1'], inc['y1'], inc['x2'], inc['y2'])

            if d_max_lvl >= 1:
                report_conf_ema = 0.35 * d_max_conf + 0.65 * report_conf_ema
            else:
                report_conf_ema = max(0.0, report_conf_ema - 0.01)

            # ---- Escalation state machine (two-phase confirmation) ----
            # Phase 1: L1 must be continuously detected for CONFIRM_SECONDS
            #          before the escalation timer even starts.
            # Phase 2: Once confirmed, escalation timer counts toward L2.
            #          If L1 drops out, timer pauses (grace period still applies).
            if d_max_lvl == 1:
                last_drowning_time = disp_now
                if d_incident:
                    incident_box = d_incident

                # Phase 1: continuous confirmation
                if confirm_start_time is None:
                    confirm_start_time = disp_now

                confirm_elapsed = disp_now - confirm_start_time
                if confirm_elapsed >= CONFIRM_SECONDS and drowning_start_time is None:
                    drowning_start_time = disp_now
                    print(f"[CONFIRMED] Drowning verified for {confirm_elapsed:.1f}s — "
                          f"escalation timer started (conf {d_max_conf:.2f})")
            else:
                # L1 absent — allow brief dropouts before breaking confirmation
                if confirm_start_time is not None:
                    dropout_s = disp_now - (last_drowning_time or disp_now)
                    if dropout_s >= CONFIRM_DROPOUT_S:
                        confirm_start_time = None

                if last_drowning_time is not None \
                        and (disp_now - last_drowning_time) >= GRACE_TIME:
                    print("[ESCALATION] Reset — drowning absent for grace period")
                    _reset_state()

            alert_level       = 0
            drowning_duration = 0.0
            if drowning_start_time is not None:
                drowning_duration = disp_now - drowning_start_time
                if drowning_duration >= ESCALATION_TIME:
                    alert_level = 2
                    if not level2_triggered:
                        level2_triggered    = True
                        level2_trigger_time = disp_now
                        send_telegram_alert()
                        _log_event(2, 'alarm', report_conf_ema * 100)
                    elif level2_trigger_time and (disp_now - level2_trigger_time) >= RESPONSE_COUNTDOWN:
                        # Response window expired — auto-reset so monitoring restarts fresh
                        print("[ESCALATION] Response window expired — resetting for new detection cycle")
                        _reset_state()
                else:
                    alert_level = 1
                    _log_event(1, 'beep', report_conf_ema * 100)

            current_alert_level = alert_level
            draw_boxes(display_frame, d_drawlist, alert_level, incident_box)
            draw_banner(display_frame, alert_level, report_conf_ema * 100,
                        drowning_duration, level2_trigger_time)
            _, buf = cv2.imencode('.jpg', display_frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

    cap.release()
    print("Stream ended")

# ======================== ROUTES ========================
@app.route('/')
def index():
    return render_template('dashboard_live.html')

@app.route('/video_feed')
def video_feed():
    global current_source
    if current_source is not None:
        return Response(generate_frames(current_source),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response("No source selected", status=400)

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global current_source, detection_active
    _reset_state()
    current_source   = 0
    detection_active = True
    return jsonify({"status": "success", "message": "Webcam started"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active, current_source
    detection_active = False
    current_source   = None
    _reset_state()
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_source, detection_active
    detection_active = False
    current_source   = None
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400
    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
    try:
        filename = secure_filename(file.filename) or f"upload_{int(time.time())}.mp4"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return jsonify({"status": "error", "message": "File saved but appears empty"}), 500
        _reset_state()
        current_source   = filepath
        detection_active = True
        return jsonify({"status": "success",
                        "message": f"Video uploaded: {filename}",
                        "filename": filename})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Upload failed: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"status": "error", "message": "File too large (max 500 MB)"}), 413

@app.route('/get_stats', methods=['GET'])
def get_stats():
    return jsonify({
        "active":      detection_active,
        "fps":         round(current_fps, 1),
        "alert_level": current_alert_level,
    })

@app.route('/get_detections', methods=['GET'])
def get_detections():
    global latest_detections
    out = latest_detections.copy()
    latest_detections.clear()
    if not out:
        return jsonify({"status": "success", "detections": []})

    # Aggregate all queued events into a single summary per poll.
    # Return only the highest-level event with mean confidence across all events at that level.
    # This ensures the frontend always receives one stable averaged data point per second,
    # never a burst of fluctuating raw per-frame values.
    max_lvl    = max(e['level'] for e in out)
    lvl_events = [e for e in out if e['level'] == max_lvl]
    avg_conf   = round(sum(e['confidence'] for e in lvl_events) / len(lvl_events), 2)
    summary = {
        'type':       lvl_events[-1]['type'],
        'level':      max_lvl,
        'audio':      lvl_events[-1]['audio'],
        'confidence': avg_conf,
        'timestamp':  lvl_events[-1]['timestamp'],
        'count':      len(out),   # diagnostic: how many raw events were merged
    }
    return jsonify({"status": "success", "detections": [summary]})

# ======================== MAIN ========================
if __name__ == '__main__':
    mdl  = _metrics.get('current_model', 'best.pt')
    n    = len(model.names) if hasattr(model, 'names') else '?'
    mAP  = _metrics.get('training', {}).get('mAP50', 'N/A')
    prec = _metrics.get('training', {}).get('precision', 'N/A')
    rec  = _metrics.get('training', {}).get('recall', 'N/A')
    print(f"\n  Dashboard → http://localhost:5000")
    print(f"  Model   : {mdl} | {n} classes | mAP50={mAP}  prec={prec}  recall={rec}")
    print(f"  L1 gate : conf >= {DROWNING_CONF_MIN}")
    print(f"  Confirm : {CONFIRM_SECONDS}s continuous L1 → escalation timer ({ESCALATION_TIME}s to L2)\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
