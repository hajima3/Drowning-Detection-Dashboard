# YOLOv11 Drowning Detection Dashboard

Real-time pool safety monitoring with 2-level alert system.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place your trained model:**
   - Export `best.pt` from Label Studio
   - Replace the `best.pt` file in the root folder
   - Ensure model meets 95-98% F1 score target

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
> (`model_manager.py`, `model_metrics.json`). This is for backend use only and
> is not shown in the dashboard UI.

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
