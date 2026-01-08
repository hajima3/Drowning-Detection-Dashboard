# 🏊 YOLOv11 Drowning Detection Dashboard

**Real-Time AI-Powered Pool Safety Monitoring System**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.3.0+-green.svg)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A sophisticated drowning detection system using YOLOv11 computer vision to monitor swimming pools in real-time. Features a beautiful web dashboard with automated incident logging, multi-level alerts, and comprehensive analytics.

---

## 🌟 Key Features

### 🎯 Core Detection Capabilities
- **Real-Time Webcam Detection** - Live monitoring at 25-30 FPS
- **Video File Analysis** - Process uploaded MP4, AVI, MOV videos
- **2-Level Alert System**:
  - 🚨 **Level 2 Emergency**: 65%+ confidence, 3+ second duration, critical drowning
  - ⚠️ **Level 1 Warning**: 50-64% confidence, unsafe movement patterns
- **YOLOv11n Model**: 96.92% mAP50 accuracy with 2.59M parameters
- **Classes**: Drowning, Swimming

### 📊 Dashboard & Logging
- **Glassmorphism UI** with animated koi fish background
- **Incident Logging System**: Automatic localStorage-based CRUD operations
- **Statistics Dashboard**: Real-time metrics (Level 2/1 counts, daily incidents, totals)
- **CSV Export**: Download complete incident reports
- **Filter Options**: All, Level 2 Only, Level 1 Only, Today
- **Editable Notes**: Add context to each incident log

### ⚡ Performance Optimizations
- Frame skipping (process every 2nd frame)
- Resolution scaling (50% default)
- DirectShow backend for Windows webcam
- JPEG compression (65% quality)
- Configurable performance presets

### 🔔 Alert Features
- Browser notifications with different behaviors for alert levels
- Audio alerts (more urgent for Level 2)
- Visual pulsing animations
- Duration tracking (prevents duplicate logs)

---

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11 (optimized with DirectShow backend)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~2GB for dependencies and model files
- **Webcam**: Any USB or built-in camera (for live detection)
- **GPU**: Optional, NVIDIA GPU with CUDA for faster processing

### Software Dependencies
- **Python**: 3.8 or higher
- **pip**: Latest version recommended

See `requirements.txt` for complete Python package list.

---

## 🚀 Quick Start

### 1. Clone/Download Repository
```bash
# Clone the repository (if using git)
git clone https://github.com/yourusername/yolov11dashboard.git
cd yolov11dashboard

# Or extract downloaded ZIP file
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Obtain Model File
⚠️ **CRITICAL**: You need the `best.pt` model file (5-10 MB)

**Option A** - Download pre-trained model:
- Check repository releases for `best.pt` download link
- Place `best.pt` in the project root directory

**Option B** - Train your own model:
- See `MODEL_README.md` for training instructions
- Requires YOLOv11 training dataset with drowning/swimming annotations

### 4. Run the Dashboard
```bash
# Using the PowerShell script (Windows)
.\scripts\start_dashboard.ps1

# Or manually
python app.py
```

### 5. Open Dashboard
Navigate to: `http://localhost:5000`

---

## 📖 Usage Guide

### Starting Webcam Detection
1. Click **🎥 Start Webcam Detection**
2. Grant camera permissions if prompted
3. Watch live feed with real-time detection overlays
4. Incidents automatically logged to **📋 Incident Logs** tab

### Uploading Video Files
1. Click **📁 Choose Video File**
2. Select MP4, AVI, or MOV file
3. Video processes automatically with detection
4. Review logged incidents in Logs tab

### Managing Incident Logs
- **View Statistics**: Level 2 emergencies, Level 1 warnings, daily/total counts
- **Filter Logs**: Click filter buttons (All, Level 2, Level 1, Today)
- **Edit Notes**: Click ✏️ Edit button, add context, click 💾 Save
- **Delete Logs**: Click 🗑️ Delete button (confirms before deletion)
- **Export Data**: Click 💾 Export to CSV for analysis
- **Clear All**: Click 🗑️ Clear All Logs (use with caution)

### Adjusting Performance
Edit `performance_settings.py` to configure:
- `PROCESS_EVERY_N_FRAMES` - Frame skip rate (2 = every 2nd frame)
- `SCALE_FACTOR` - Resolution scaling (0.5 = 50% size)
- `JPEG_QUALITY` - Compression quality (65 = 65% quality)
- `DEFAULT_CONFIDENCE` - Detection threshold (0.5 = 50% confidence)

---

## 🎨 Dashboard Overview

### Live Detection Tab
- **Video Feed**: Real-time or uploaded video with detection boxes
- **Status Indicator**: 🟢 Active / 🔴 Inactive
- **Camera Controls**: Start/Stop webcam
- **Upload Controls**: Video file selection

### Incident Logs Tab
- **Statistics Cards**: 
  - 🚨 Level 2 Emergencies (75%+ confidence)
  - ⚠️ Level 1 Warnings (50-75%)
  - 📅 Today's Incidents
  - 📊 Total Alerts
- **Filters**: Quick filter buttons
- **Logs Table**: ID, Timestamp, Alert Level, Confidence, Notes, Actions
- **Export Button**: Download CSV report

---

## 🛠️ Technical Details

### Model Architecture
- **Base**: YOLOv11n (Nano variant)
- **Parameters**: 2.59M
- **Accuracy**: 96.92% mAP50
- **Input Size**: 640x640
- **Classes**: 2 (drowning, swimming)
- **Speed**: 25-30 FPS on modern hardware

### Detection Logic
```python
# Level 2 Emergency Triggers:
- Confidence ≥ 80% (critical drowning behavior)
- Duration ≥ 3 seconds (submerged/struggle)
- Confidence ≥ 65% (erratic movement patterns)

# Level 1 Warning Triggers:
- Confidence 50-64% (unsafe movement)
- Monitoring required
```

### Performance Tuning
| Preset | Frame Skip | Scale | Quality | Use Case |
|--------|------------|-------|---------|----------|
| Real-time | 2 | 0.5 | 65% | Live webcam, smooth |
| High-Quality | 1 | 0.75 | 85% | Video analysis |
| Maximum Speed | 3 | 0.4 | 50% | Low-end systems |

---

## 📂 Project Structure

```
yolov11dashboard/
├── app.py                      # Flask server with detection logic
├── performance_settings.py     # Configuration parameters
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusion patterns
├── best.pt                     # YOLOv11 model weights (not included)
├── templates/
│   └── dashboard_live.html     # Web dashboard UI
├── uploads/                    # Uploaded video storage
│   └── .gitkeep
├── scripts/
│   └── start_dashboard.ps1     # Windows launcher script
└── docs/
    ├── README.md               # This file
    ├── SETUP_GUIDE.md          # Detailed installation
    ├── QUICK_REFERENCE.md      # Commands & shortcuts
    ├── CHANGELOG.md            # Version history
    ├── PROJECT_INFO.md         # Technical specifications
    └── MODEL_README.md         # Model acquisition guide
```

---

## 🐛 Troubleshooting

### Issue: "Model file 'best.pt' not found"
**Solution**: Download or train the model file and place in project root
- See `MODEL_README.md` for instructions

### Issue: Webcam not detected
**Solution**: 
1. Check camera permissions in Windows Settings
2. Verify camera works in other applications
3. Try changing `CAP_DSHOW` backend in `app.py`

### Issue: Low FPS performance
**Solution**: Adjust performance settings:
```python
# In performance_settings.py
PROCESS_EVERY_N_FRAMES = 3  # Skip more frames
SCALE_FACTOR = 0.4          # Reduce resolution
```

### Issue: Detection inaccurate
**Solution**:
1. Ensure good lighting conditions
2. Adjust `DEFAULT_CONFIDENCE` threshold
3. Retrain model with more diverse dataset

For more issues, see `QUICK_REFERENCE.md` troubleshooting section.

---

## 📚 Additional Documentation

- **📘 [SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed installation instructions
- **📙 [QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Commands, shortcuts, common fixes
- **📗 [PROJECT_INFO.md](PROJECT_INFO.md)** - Technical architecture details
- **📕 [CHANGELOG.md](CHANGELOG.md)** - Version history and updates
- **📖 [MODEL_README.md](MODEL_README.md)** - Model acquisition and training

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ultralytics YOLOv11** - State-of-the-art object detection framework
- **Flask** - Lightweight web framework
- **OpenCV** - Computer vision library
- **PyTorch** - Deep learning backend

---

## ⚠️ Safety Disclaimer

**IMPORTANT**: This system is a **supplementary safety tool**, not a replacement for:
- Professional lifeguards
- Constant human supervision
- Proper pool safety protocols
- Emergency response procedures

**Always maintain direct human oversight of swimming areas.**

---

## 📞 Support

For issues, questions, or suggestions:
- Open a GitHub issue
- Check documentation files
- Review troubleshooting section

---

**Built with ❤️ for Pool Safety | Powered by YOLOv11**
