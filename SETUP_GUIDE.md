# 🚀 Setup Guide - YOLOv11 Drowning Detection Dashboard

Complete step-by-step setup instructions for deploying the dashboard on any device.

## 📋 Prerequisites

Before you begin, ensure you have:

- ✅ **Python 3.8+** installed ([Download Python](https://www.python.org/downloads/))
- ✅ **Git** (optional, for cloning) ([Download Git](https://git-scm.com/downloads))
- ✅ **8GB+ RAM** (16GB recommended)
- ✅ **Webcam** (built-in or USB)
- ✅ **Modern browser** (Chrome, Edge, Firefox)

## 🔧 Installation Steps

### Step 1: Download/Clone the Repository

**Option A: Download ZIP**
1. Download the project ZIP file
2. Extract to your desired location (e.g., `C:\YOLOv11-Dashboard`)

**Option B: Git Clone**
```bash
git clone <repository-url>
cd Final-YOLOv11-Dashboard
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected packages:**
- ultralytics (YOLOv11)
- opencv-python (Video processing)
- flask (Web server)
- torch, torchvision (Deep learning)
- numpy, matplotlib (Data processing)

**Installation time**: 5-15 minutes depending on internet speed

### Step 4: Verify Model File

Ensure `best.pt` is present in the project root:

```
Final YOLOv11 Dashboard/
├── app.py
├── best.pt  ← This file should exist (trained model)
├── ...
```

**Model details:**
- File size: ~5-10 MB
- Type: YOLOv11n (Nano)
- Accuracy: 96.92% mAP50

### Step 5: Test Installation

Run a quick test:

```bash
python -c "import ultralytics; import cv2; import flask; print('✅ All packages installed successfully!')"
```

## 🎯 First Launch

### Starting the Server

**Method 1: Python Command**
```bash
python app.py
```

**Method 2: PowerShell Script (Windows)**
```powershell
.\scripts\start_dashboard.ps1
```

### Expected Output

```
✅ Loaded custom performance settings
Loading model: best.pt
✅ Model loaded successfully!

============================================================
🌊 YOLOv11 Drowning Detection Web Dashboard
============================================================

📡 Starting server...
🌐 Open browser at: http://localhost:5000

✅ Ready to detect!

* Serving Flask app 'app'
* Debug mode: on
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5000
* Running on http://192.168.X.X:5000
```

### Accessing the Dashboard

1. Open your browser
2. Navigate to: **http://localhost:5000**
3. You should see the dashboard with animated koi fish background

## 🧪 Testing the System

### Test 1: Webcam Detection

1. Click **"🎥 Start Webcam Detection"**
2. Allow camera permissions
3. Wave your hand to test detection
4. Click **"⏹️ Stop Detection"** when done

### Test 2: Manual Log Entry

Open browser console (F12) and run:

```javascript
// Test Level 1 warning
testLevel1()

// Test Level 2 emergency
testLevel2()
```

Check the **"📋 Incident Logs"** tab to see test entries.

### Test 3: Video Upload

1. Prepare a test video (MP4, AVI, or MOV)
2. Click **"📁 Choose Video File"**
3. Select your video
4. Watch for automatic detections

## ⚙️ Configuration

### Adjusting Performance

Edit `performance_settings.py`:

**For slower computers:**
```python
PROCESS_EVERY_N_FRAMES = 3  # Skip more frames
SCALE_FACTOR = 0.5          # Lower resolution
JPEG_QUALITY = 60           # Lower streaming quality
```

**For faster computers:**
```python
PROCESS_EVERY_N_FRAMES = 1  # Process every frame
SCALE_FACTOR = 0.75         # Higher resolution
JPEG_QUALITY = 85           # Higher streaming quality
```

### Changing Port

Edit `app.py` (bottom of file):

```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Change to 8080
```

### Detection Thresholds

In `app.py`, modify these values:

```python
LEVEL_2_DURATION_THRESHOLD = 3.0  # Seconds before Level 2

# In detection logic:
if conf_percentage >= 80:      # Critical confidence
if conf_percentage >= 65:      # Erratic movement
if drowning_duration >= 3.0:   # Duration threshold
```

## 🌐 Network Access

### Accessing from Other Devices

1. Find your computer's IP address:

**Windows:**
```powershell
ipconfig
# Look for IPv4 Address (e.g., 192.168.1.100)
```

**macOS/Linux:**
```bash
ifconfig
# Look for inet address
```

2. On another device (same network), navigate to:
```
http://YOUR-IP-ADDRESS:5000
```

### Firewall Configuration

**Windows Firewall:**
1. Open Windows Defender Firewall
2. Allow Python through firewall
3. Allow port 5000 (TCP)

## 🔒 Security Recommendations

### For Production Use

1. **Disable debug mode** in `app.py`:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

2. **Use production server**:
```bash
pip install gunicorn  # Linux/macOS
pip install waitress   # Windows

# Run with waitress
waitress-serve --port=5000 app:app
```

3. **Add authentication**:
```python
from flask_httpauth import HTTPBasicAuth
```

4. **Enable HTTPS** with SSL certificates

5. **Restrict host**:
```python
app.run(host='127.0.0.1', port=5000)  # Localhost only
```

## 🐛 Common Issues

### Issue: "Module not found"
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue: "Camera not accessible"
**Solution:**
- Close other apps using camera
- Check camera permissions
- Restart browser
- Try different browser

### Issue: "Port already in use"
**Solution:**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9
```

### Issue: "Model file not found"
**Solution:**
- Verify `best.pt` exists in project root
- Re-download model if corrupted
- Check file permissions

### Issue: "Low FPS / Lagging"
**Solution:**
- Lower `SCALE_FACTOR` to 0.5
- Increase `PROCESS_EVERY_N_FRAMES` to 3
- Close unnecessary applications
- Use GPU if available

## 📊 System Requirements Check

Run this diagnostic script:

```python
import sys
import cv2
import torch
import ultralytics

print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Ultralytics version: {ultralytics.__version__}")

# Test webcam
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Webcam accessible")
    cap.release()
else:
    print("❌ Webcam not accessible")
```

## 🎓 Next Steps

After successful setup:

1. ✅ Read full documentation in `README.md`
2. ✅ Test with sample drowning videos
3. ✅ Adjust confidence thresholds for your use case
4. ✅ Configure alert system (Level 1 vs Level 2)
5. ✅ Set up incident log exports
6. ✅ Train staff on alert response protocols

## 📞 Getting Help

If you encounter issues:

1. Check the **Troubleshooting** section in `README.md`
2. Review **Common Issues** above
3. Check Python/package versions
4. Verify model file integrity
5. Test with simplified settings

## 📦 Deployment Checklist

Before deploying to production:

- [ ] All dependencies installed
- [ ] Model file verified (best.pt)
- [ ] Webcam tested successfully
- [ ] Video upload tested
- [ ] Alert system tested (Level 1 & 2)
- [ ] Incident logs working
- [ ] CSV export functional
- [ ] Browser notifications enabled
- [ ] Performance optimized for hardware
- [ ] Firewall configured (if needed)
- [ ] Staff trained on system usage
- [ ] Emergency response protocols established

## 🎉 Success!

Your YOLOv11 Drowning Detection Dashboard is now ready to use!

**Dashboard**: http://localhost:5000

**Features available:**
- ✅ Real-time webcam detection
- ✅ Video file upload
- ✅ 2-Level alert system
- ✅ Automatic incident logging
- ✅ Filter & export logs
- ✅ Browser notifications

---

**⚠️ Remember**: This is a supplementary safety tool. Always maintain proper lifeguard supervision!
