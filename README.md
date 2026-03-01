# Drowning Detection Dashboard

Real-time pool safety monitoring system using YOLOv11 object detection, compliant with the **10/20 Protection Standard** (detect within 10 seconds, respond within 20 seconds).

---

## Requirements

- Python 3.10+
- Conda environment (`env2`)
- NVIDIA GPU recommended (CUDA / FP16 inference)
- Webcam or video file input

Install dependencies:

```powershell
conda activate env2
pip install -r requirements.txt
```

---

## Quick Start

```powershell
conda activate env2
python app.py
```

Open the dashboard at: `http://localhost:5000`

---

## Alert Levels

| Level                      | Trigger                                                            | Display                     | Audio        | Telegram                 |
| -------------------------- | ------------------------------------------------------------------ | --------------------------- | ------------ | ------------------------ |
| **0 — Safe**               | Swimming or floating detected                                      | Blue bounding box           | Silent       | No                       |
| **1 — Drowning Detected**  | Drowning at >= 40% confidence, confirmed for 2 continuous seconds  | Orange box + banner         | Beep         | No                       |
| **2 — Emergency**          | Level 1 sustained for 5 additional seconds                         | Red box + emergency banner  | Alarm once   | Yes — once per incident  |

A 3-second grace period allows brief detection dropouts before resetting the escalation timer.
After the 20-second response window expires, the system auto-resets and resumes monitoring.

---

## Model

| Property         | Value                       |
| ---------------- | --------------------------- |
| File             | `best.pt`                   |
| Architecture     | YOLOv11-L                   |
| Parameters       | 25,281,625                  |
| Layers (fused)   | 191                         |
| GFLOPs           | 86.6                        |
| Classes          | swimming, floating, drowning|
| Training epochs  | 200 (best at epoch 189)     |
| mAP50            | 99.21%                      |
| mAP50-95         | 84.47%                      |
| Precision        | 98.14%                      |
| Recall           | 98.77%                      |
| Input size       | 640 x 640                   |
| Inference        | FP16 half-precision (GPU)   |
| Deployed         | 2026-03-01                  |

---

## Telegram Alerts

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

If credentials are not set, Telegram alerts are skipped silently.

---

## Audio Files

Place the following files in `static/audio/`:

| File           | Usage                                          |
| -------------- | ---------------------------------------------- |
| `level1.mp3`   | Short beep — played on every Level 1 detection |
| `level2.mp3`   | Alarm — played once when Level 2 fires         |

---

## Configuration

All thresholds are defined in `detection_config.json`. Key values:

| Parameter              | Default  | Description                                                     |
| ---------------------- | -------- | --------------------------------------------------------------- |
| `confidence`           | 0.25     | YOLO inference gate — detections below this are discarded       |
| `drowning_conf_min`    | 0.40     | Minimum confidence to classify a detection as drowning          |
| `confirm_seconds`      | 2.0 s    | Continuous Level 1 required before escalation timer starts      |
| `escalation_time_s`    | 5.0 s    | Sustained Level 1 after confirmation before Level 2 fires       |
| `grace_time_s`         | 3.0 s    | Dropout tolerance before escalation resets                      |
| `display_delay_s`      | 1.0 s    | Display lag used as temporal lookahead buffer                   |
| `target_fps`           | 30       | Target capture frame rate                                       |

Settings can be overridden at runtime via `.env` environment variables.

---

## Project Structure

```text
app.py                  Main Flask application
best.pt                 YOLOv11-L trained model
detection_config.json   Detection and alert thresholds
class_mapping.json      Class-to-alert-level mapping
model_metrics.json      Training metrics and class weights
requirements.txt        Python dependencies
static/audio/           Alert sound files
templates/              HTML dashboard
uploads/                Temporary uploaded video files
```
