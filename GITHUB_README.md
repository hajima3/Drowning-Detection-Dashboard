# 🏊 YOLOv11 Drowning Detection Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00D4AA.svg)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-2.3.0+-black.svg)](https://flask.palletsprojects.com/)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.92%25-success.svg)](#model-performance)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)](#license)

A **real-time AI-powered drowning detection system** using YOLOv11 with a comprehensive web dashboard for pool safety monitoring. Features a sophisticated 2-level alert system, automatic incident logging, and beautiful animated UI.

---

## 🌟 Key Features

### 🎯 Detection System
- ✅ **96.92% accuracy** using YOLOv11n (Nano) model
- ✅ **Real-time webcam detection** at 25-30 FPS (640x480)
- ✅ **Video file support** (MP4, AVI, MOV) with 4K handling
- ✅ **Multi-factor analysis**: Confidence + Duration + Behavior
- ✅ **Duration tracking**: 3-second threshold for emergencies

### 🚨 2-Level Alert System
- **Level 1 (⚠️ Warning)**: 50-64% confidence - Unsafe movement detected
- **Level 2 (🚨 Emergency)**: Drowning detected via:
  - 80%+ critical confidence (instant)
  - 65%+ erratic movement patterns
  - 3+ seconds continuous submersion

### 📊 Dashboard Features
- 📹 **Live video feed** with real-time detection overlay
- 🎛️ **Adjustable confidence threshold** (10-90%)
- 📈 **Statistics dashboard** with 4 real-time metrics
- 📋 **Automatic incident logging** with timestamps
- 🔍 **Advanced filtering** (All, Level 2, Level 1, Today)
- 💾 **CSV export** for record-keeping
- ✏️ **Editable notes** per incident
- 🔔 **Browser notifications** (dismissible Level 1, persistent Level 2)
- 🎨 **Beautiful UI** with animated koi fish background

---

## 📸 Screenshots

*(Add screenshots here if desired)*

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (built-in or USB)
- 8GB RAM minimum (16GB recommended)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yolov11-drowning-detection-dashboard.git
   cd yolov11-drowning-detection-dashboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\Activate.ps1
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the dashboard**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

**That's it!** 🎉 Start detecting with webcam or upload a video.

---

## 📖 Usage

### Starting Webcam Detection
1. Click **"🎥 Start Webcam Detection"**
2. Allow camera permissions if prompted
3. Watch real-time detection with automatic logging

### Uploading Video Files
1. Click **"📁 Choose Video File"**
2. Select video (MP4, AVI, MOV)
3. Video processes automatically with detection

### Managing Incident Logs
1. Switch to **"📋 Incident Logs"** tab
2. View all alerts with timestamp, level, confidence
3. Filter by: All, Level 2, Level 1, or Today
4. Edit notes, delete logs, or export to CSV

### Adjusting Sensitivity
- Use the **confidence threshold slider** (10-90%)
- Lower = More detections (may include false positives)
- Higher = Only very confident detections

---

## 🎯 Alert System Explained

| Alert Level | Confidence | Behavior | Response |
|-------------|-----------|----------|----------|
| **Level 1** ⚠️ | 50-64% | Monitoring | Staff observation recommended |
| **Level 2** 🚨 | 65%+ OR 80%+ OR 3s+ | Emergency | Immediate intervention required |

### Level 2 Triggers
1. **Critical confidence**: 80%+ → Instant Level 2
2. **Erratic movement**: 65%+ with irregular patterns
3. **Duration**: 3+ seconds continuous submersion

---

## 📊 Model Performance

- **Architecture**: YOLOv11n (Nano)
- **Parameters**: 2.59 Million
- **mAP50**: 96.92%
- **Precision**: 90.79%
- **Recall**: 91.86%
- **Classes**: 2 (drowning, swimming)
- **Training**: 50 epochs on custom dataset

---

## ⚙️ Configuration

### Performance Tuning

Edit `performance_settings.py` to customize:

```python
# Frame processing
PROCESS_EVERY_N_FRAMES = 1  # 1 = smoothest, 3 = fastest

# Resolution scaling
SCALE_FACTOR = 0.6  # 0.5-1.0 (lower = faster)

# Streaming quality
JPEG_QUALITY = 75  # 50-95 (lower = faster)

# Default confidence
DEFAULT_CONFIDENCE = 0.5  # 0.1-0.9
```

**Quick presets:**
- **Smooth webcam**: `FRAMES=1, SCALE=0.6, QUALITY=75`
- **Maximum speed**: `FRAMES=3, SCALE=0.5, QUALITY=60`
- **Best quality**: `FRAMES=1, SCALE=0.75, QUALITY=85`

---

## 📁 Project Structure

```
yolov11-drowning-detection-dashboard/
├── app.py                      # Flask server with detection logic
├── performance_settings.py     # Performance configuration
├── requirements.txt            # Python dependencies
├── best.pt                     # Trained YOLOv11n model
├── templates/
│   └── dashboard_live.html    # Web dashboard UI
├── uploads/                    # Video upload directory
├── scripts/
│   └── start_dashboard.ps1    # Quick start script (Windows)
└── docs/
    ├── README.md              # This file
    ├── SETUP_GUIDE.md         # Detailed installation guide
    ├── QUICK_REFERENCE.md     # Commands and shortcuts
    ├── CHANGELOG.md           # Version history
    └── PROJECT_INFO.md        # Technical specifications
```

---

## 🔧 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 | Windows 11 |
| **Python** | 3.8 | 3.10+ |
| **RAM** | 8GB | 16GB |
| **Storage** | 3GB | 5GB |
| **GPU** | Optional | CUDA-compatible |
| **Webcam** | 640x480 | 1080p |

---

## 🧪 Testing

### Test Level 1 Warning
Open browser console (F12) and run:
```javascript
testLevel1()
```

### Test Level 2 Emergency
```javascript
testLevel2()
```

Both functions create sample log entries for testing the incident management system.

---

## 🐛 Troubleshooting

### Camera Not Working
- Close other apps using the camera
- Allow camera permissions in browser
- Try refreshing the page (Ctrl + F5)

### Low FPS / Lagging
- Lower `SCALE_FACTOR` to 0.5 in `performance_settings.py`
- Increase `PROCESS_EVERY_N_FRAMES` to 3
- Close unnecessary applications

### Level 2 Not Triggering
- Lower confidence threshold to 50%
- Verify 3-second duration is enabled
- Check console logs for detection events

For more solutions, see **[SETUP_GUIDE.md](SETUP_GUIDE.md)** troubleshooting section.

---

## 📚 Documentation

- **[README.md](README.md)** - Project overview (you are here)
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete installation guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Commands and features
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[PROJECT_INFO.md](PROJECT_INFO.md)** - Technical specifications

---

## 🔐 Security Notice

This is a **development server** using Flask's built-in server.

### For Production:
- Use WSGI server (gunicorn/waitress)
- Enable HTTPS/SSL
- Add authentication
- Use database backend
- Implement user management

See **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for production deployment instructions.

---

## 🚨 Safety Disclaimer

**IMPORTANT**: This system is a **supplementary safety tool** only.

### ✅ System Should:
- Enhance existing safety measures
- Provide early warning alerts
- Log incidents for review
- Assist trained personnel

### ❌ System Should NOT:
- Replace human lifeguards
- Be the sole safety measure
- Operate without supervision
- Guarantee 100% detection

**Always maintain proper lifeguard supervision and follow standard pool safety protocols.**

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

This project is provided **as-is** for educational and safety monitoring purposes.

**Note**: This is a supplementary safety tool. It should enhance, not replace, human vigilance and proper safety protocols.

---

## 🙏 Acknowledgments

- **[Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)** - Object detection framework
- **[Flask](https://flask.palletsprojects.com/)** - Web framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- Custom drowning detection dataset

---

## 📊 Statistics

- **⭐ Accuracy**: 96.92% mAP50
- **🚀 FPS**: 25-30 (webcam), real-time (video)
- **📦 Model Size**: ~5-10 MB
- **💾 Total Package**: ~10-15 MB (without dependencies)
- **🎯 Detection Classes**: 2 (drowning, swimming)

---

## 📞 Support

For issues, questions, or feature requests:
- Open an [issue](https://github.com/yourusername/repo/issues)
- Check [documentation](SETUP_GUIDE.md)
- Review [troubleshooting](SETUP_GUIDE.md#troubleshooting)

---

## 🎉 Quick Links

- 🚀 [Quick Start](#quick-start)
- 📖 [Usage Guide](#usage)
- ⚙️ [Configuration](#configuration)
- 🔧 [Troubleshooting](#troubleshooting)
- 📚 [Full Documentation](SETUP_GUIDE.md)
- 📊 [Model Performance](#model-performance)
- 🔐 [Security](SETUP_GUIDE.md#security-recommendations)

---

**Version**: 1.0.0  
**Release Date**: October 2025  
**Status**: Production Ready ✅  

Made with ❤️ for pool safety

---

<div align="center">

**⚠️ Remember: AI enhances safety, but never replaces human supervision! ⚠️**

</div>
