# YOLOv11 Drowning Detection Dashboard

Real-time pool safety monitoring with a 3-level behavior system (Normal / Warning / Emergency).

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your trained model:**
   - Export `best.pt` from Label Studio
   - Replace the `best.pt` file in the root folder
   - Ensure model meets 95-98% F1 score target
   - Optional: set `MODEL_PATH` to load a model from a different location

3. **Add audio alert files** (optional):
   - `static/audio/level1.mp3` - Level 1 warning sound
   - `static/audio/level2.mp3` - Level 2 emergency sound

4. **Run the dashboard:**
   ```bash
   python app.py
   ```
   Or use the PowerShell script:
   ```bash
   .\scripts\start_dashboard.ps1
   ```

5. **Open browser:** http://localhost:5000

## Model Performance Requirements

- **Precision**: Minimize false alarms (false positives)
- **Recall**: Detect all real drowning cases (minimize misses)
- **F1 Score Target**: 95-98% for production deployment
- If your model is below 95%, retrain with more data in Label Studio

## Features

- **Real-time webcam detection** - Live pool monitoring
- **Video file analysis** - Upload and analyze videos
- **3-Level Alert System**:
  - Level 0: Normal swimming behavior (19 classes)
  - Level 1: Unsafe/erratic movement (42 classes) - Orange warning
  - Level 2: Critical drowning emergency (21 classes) - Red alert
- **60+ Behavior Classes** - Comprehensive classification system
- **Automatic class mapping** - Classes automatically trigger correct alert level
- **Incident logging** - Automatic detection logging with timestamps
- **Audio alerts** - Play different sounds for Level 1 and Level 2
  
> Advanced: You can optionally track model iterations and accuracy using the CLI tools
> (`model_manager.py`, `model_metrics.json`). The dashboard backend exposes `/get_model_metrics`,
> but the current UI does not render model metrics.

## How We Built This Dashboard (End-to-End)

This project is a single Flask web app that:

1. Captures frames from a webcam or uploaded video (OpenCV)
2. Runs YOLOv11 inference on selected frames (Ultralytics)
3. Converts detected classes into alert levels (mapping rules)
4. Streams annotated frames to the browser as an MJPEG feed
5. Sends alert events to the UI for logging + audio + optional Telegram Level 2 notifications

### High-Level Architecture

- **Backend (Python/Flask)**: [app.py](app.py)
   - Serves the web UI (`/`)
   - Streams live annotated video (`/video_feed`)
   - Controls sources (`/start_webcam`, `/upload_video`, `/stop_detection`)
   - Accepts confidence updates (`/set_confidence`)
   - Exposes detection events (`/get_detections`)
   - Optional integrations: Telegram bot alerts (Level 2 only)

- **Frontend (HTML/CSS/JS)**: [templates/dashboard_live.html](templates/dashboard_live.html)
   - Starts webcam or uploads a video
   - Displays the MJPEG stream
   - Polls the backend every second for new detection events
   - Logs incidents to the browser’s `localStorage` and supports CSV export
   - Plays audio on Level 1 and Level 2

### Backend Flow (What Happens When Detection Runs)

1. **Model load on startup**
    - On app start, Ultralytics loads the YOLO model from `best.pt`.

2. **Start a source**
    - Webcam: UI calls `POST /start_webcam` → backend sets `current_source = 0` and `detection_active = True`.
    - Video: UI calls `POST /upload_video` → backend saves the file into `uploads/` and sets `current_source` to that file path.

3. **Browser opens the video stream**
    - UI injects an `<img>` tag pointing to `/video_feed`.
    - Flask returns a `multipart/x-mixed-replace` (MJPEG) stream.

4. **Frame processing + inference**
    - OpenCV reads frames from the source.
    - Performance tuning happens here:
       - `PROCESS_EVERY_N_FRAMES` skips frames to reduce load
       - `SCALE_FACTOR` runs inference at lower resolution
       - `JPEG_QUALITY` compresses the outgoing stream
    - Ultralytics YOLO runs inference with the current `confidence_threshold`.

5. **Alert level mapping**
    - Each detected class name is mapped into an alert level using:
       - `class_mapping.json` (60+ behavior classes), and
       - a fallback for simple models that only output labels like `drowning` or `swimming`.

6. **Event generation + notifications**
    - When Level 1 or Level 2 is detected, the backend creates a `detection_event` containing:
       - level, type, class, confidence, timestamp, duration (for Level 2), etc.
    - Level 2 events optionally trigger Telegram via `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`.

7. **UI polling and incident logs**
    - The browser calls `GET /get_detections` every second.
    - The backend returns and clears the “latest detections” buffer.
    - The UI:
       - saves incidents in `localStorage`
       - plays `/static/audio/level1.mp3` or `/static/audio/level2.mp3`
       - shows a prominent on-screen alert and browser notification for Level 2

### Frontend Flow (What the Page Does)

- **Live Detection tab**
   - Buttons call `/start_webcam` or `/upload_video`.
   - When a source starts, the page sets the video element to `/video_feed`.
   - Starts 1-second polling for `/get_detections`.

- **Incident Logs tab**
   - Logs are stored client-side (browser `localStorage`), not in a server database.
   - Supports filtering, editing notes, deleting entries, and exporting to CSV.

### Optional Telegram Setup (Level 2 Only)

Create a `.env` file in the project root (or set environment variables) with:

```
TELEGRAM_BOT_TOKEN=123456:ABCDEF...
TELEGRAM_CHAT_ID=123456789
```

If these variables are not set, Telegram sending is automatically skipped.

## Settings

Edit settings in `app.py`:
- `PROCESS_EVERY_N_FRAMES` - Frame skip for performance
- `SCALE_FACTOR` - Resolution scaling (0.5 = 50%)
- `JPEG_QUALITY` - Stream compression quality
- `DEFAULT_CONFIDENCE` - Detection threshold
- `LEVEL_2_DURATION_THRESHOLD` - Seconds for Level 2 escalation

Edit class mappings in `class_mapping.json`:
- Add/remove behavior classes
- Assign classes to alert levels (0, 1, 2)
- See `CLASS_SYSTEM.md` for complete classification guide

## Project Structure

```
yolov11dashboard/
├── app.py                  # Main dashboard application
├── best.pt                 # YOLOv11 trained model
├── requirements.txt        # Python dependencies
├── templates/
│   └── dashboard_live.html # Dashboard UI
├── static/
│   └── audio/             # Alert sound files
│       ├── level1.mp3
│       └── level2.mp3
├── uploads/               # Uploaded videos
└── scripts/
    └── start_dashboard.ps1 # Quick start script
```

## What To Screenshot For Documentation

If you’re writing a “how we built it” report, these screenshots usually cover everything reviewers expect.

### 1) Repo / Folder Proof
- The project root folder in File Explorer showing the key files: `app.py`, `templates/`, `static/`, `requirements.txt`, `best.pt`.
- The `templates/` folder showing `dashboard_live.html`.
- The `static/audio/` folder showing `level1.mp3` and `level2.mp3`.
- The `scripts/` folder showing `start_dashboard.ps1`.

### 2) Backend Implementation Evidence (VS Code)
- [app.py](app.py):
   - The **SETTINGS** section (frame skipping/scaling/confidence)
   - The **model load** (`model = YOLO('best.pt')`)
   - The **frame generator** (`generate_frames`) showing OpenCV capture + YOLO inference
   - The **routes** section showing `/start_webcam`, `/upload_video`, `/video_feed`, `/get_detections`
- [class_mapping.json](class_mapping.json): show the Level 0/1/2 class lists.

### 3) Frontend Implementation Evidence (VS Code)
- [templates/dashboard_live.html](templates/dashboard_live.html):
   - The tab UI (“Live Detection”, “Incident Logs”)
   - The JavaScript functions calling backend endpoints (`/start_webcam`, `/upload_video`, `/set_confidence`, `/get_detections`)
   - The audio alert initialization pointing to `/static/audio/level1.mp3` and `/static/audio/level2.mp3`
   - The `localStorage` logging logic and CSV export function

### 4) Environment + Dependencies
- [requirements.txt](requirements.txt) showing `flask`, `ultralytics`, `opencv-python`, `torch`.
- (Optional) Terminal screenshot:
   - `pip install -r requirements.txt`
   - `python app.py`

### 5) Run Script / Automation
- [scripts/start_dashboard.ps1](scripts/start_dashboard.ps1) showing:
   - venv creation/activation
   - dependency installation
   - launching `python app.py`

### 6) Model + Training Documentation
- [MODEL_DEPLOYMENT.md](MODEL_DEPLOYMENT.md) (training + deploy steps)
- [CLASS_SYSTEM.md](CLASS_SYSTEM.md) (3-level behavior system explanation)
- (Optional) [ITERATION_PLAN.md](ITERATION_PLAN.md) and [model_manager.py](model_manager.py)

### 7) Live App Evidence (Browser)
- Dashboard home with the two tabs visible.
- Live detection running (webcam feed showing bounding boxes/overlay).
- Confidence slider changed (show the displayed %).
- A Level 1 log entry and a Level 2 log entry in the Incident Logs tab.
- CSV export download (or the “Logs exported!” success banner).

## Model Training & Deployment

### Training with Label Studio

1. **Annotate** your drowning/swimming videos in Label Studio
2. **Train** YOLOv11 model with your annotated dataset
3. **Evaluate** model performance:
   - **Precision**: Accuracy of positive predictions (minimize false positives)
   - **Recall**: Ability to detect all drowning cases (minimize false negatives)
   - **F1 Score**: Final accuracy metric (harmonic mean of precision & recall)
   - **Target**: 95-98% F1 score for production deployment

### Deploying Trained Model

1. Export trained model as `best.pt` from Label Studio
2. Replace the `best.pt` file in this dashboard directory
3. Restart the dashboard: `python app.py`
4. Model automatically loads - start detecting!

**Note**: The dashboard will display the confidence score for each detection. Adjust the confidence threshold in the dashboard UI based on your model's performance.
