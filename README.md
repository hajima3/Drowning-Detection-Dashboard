# YOLOv11 Drowning Detection Dashboard

Real-time pool safety monitoring powered by YOLOv11 with a 3-level alert system.

---

## Quick Start

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Place your trained model**
   - Put `best.pt` in the project root folder

3. **Add alert audio** (optional)
   - `static/audio/level2.mp3` — played on Level 2 drowning emergency

4. **Configure Telegram alerts** (optional)
   - Create a `.env` file:

   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

5. **Run**

   ```bash
   python app.py
   ```

   Then open: <http://localhost:5000>

---

## Alert Levels

| Level | Name | Color | Action | Audio |
| --- | --- | --- | --- | --- |
| **0** | Swimming | 🔵 Blue | No action | Silent |
| **1** | Risky | 🟠 Orange | Monitor — ready to intervene | Short beep |
| **2** | Drowning | 🔴 Maroon | Immediate emergency response | Alarm + SMS |

**Level 2 requires 3 continuous seconds** of detection before triggering the full alarm and Telegram notification (prevents false positives).

### Current Model Classes (`best.pt`)

| Model Output | Alert Level |
| --- | --- |
| `LVL 0` | Level 0 — Safe |
| `LVL 1` | Level 1 — Risky |
| `LVL2` | Level 2 — Drowning |

---

## Project Structure

```text
yolov11dashboard/
├── app.py                    # Main Flask application
├── best.pt                   # YOLOv11 trained model
├── class_mapping.json        # Class → alert level mapping + colors
├── model_metrics.json        # Deployed model info
├── detection_config.json     # Runtime tuning (FPS, confidence, etc.)
├── requirements.txt          # Python dependencies
├── templates/
│   └── dashboard_live.html   # Dashboard UI
├── static/audio/             # Alert sound files
│   └── level2.mp3            # Level 2 emergency alarm
├── uploads/                  # Uploaded video files
└── scripts/
    └── start_dashboard.ps1   # PowerShell quick-start
```

---

## Settings

All settings can be overridden via environment variables or `detection_config.json`:

| Setting | Default | Description |
| --- | --- | --- |
| `PROCESS_EVERY_N_FRAMES` | 1 | Skip frames to reduce CPU load |
| `SCALE_FACTOR` | 1.0 | Resize frames before inference (e.g. 0.5 = half) |
| `JPEG_QUALITY` | 65 | Stream compression (lower = faster) |
| `DEFAULT_CONFIDENCE` | 0.25 | Detection confidence threshold |
| `LEVEL_2_DURATION_THRESHOLD` | 3.0 | Seconds of Level 2 before alarm fires |
| `YOLO_DEVICE` | auto | `cpu`, `cuda:0`, or `auto` |
| `YOLO_IMGSZ` | 640 | Inference image size |

---

## How It Works

1. **Capture** — OpenCV reads frames from webcam or uploaded video
2. **Infer** — YOLOv11 runs on each frame and outputs class + confidence
3. **Map** — Detected class is mapped to Level 0 / 1 / 2 via `class_mapping.json`
4. **Alert** — Level 1 triggers a beep; Level 2 (≥ 3s) triggers alarm + Telegram
5. **Stream** — Annotated frames are streamed to the browser as MJPEG

---

## Retraining the Model

When retraining with Label Studio:

1. Annotate with bounding boxes using class names: `LVL 0`, `LVL 1`, `LVL2`
2. Train: `yolo train data=dataset.yaml model=yolov11l.pt epochs=100 imgsz=640`
3. Copy `runs/detect/train/weights/best.pt` → project root
4. Restart the dashboard
