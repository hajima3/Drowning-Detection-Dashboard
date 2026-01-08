# 📚 Quick Reference Guide

## 🚀 Starting the Dashboard

```bash
# Activate virtual environment (if using one)
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # macOS/Linux

# Start server
python app.py

# Access dashboard
http://localhost:5000
```

## 🎮 Dashboard Controls

### Detection Tab
- **Start Webcam** → Begin real-time detection
- **Stop Detection** → End current detection session
- **Choose Video File** → Upload and analyze video
- **Confidence Slider** → Adjust detection sensitivity (10-90%)

### Logs Tab
- **All Alerts** → Show all logged incidents
- **Level 2 Only** → Filter emergency alerts
- **Level 1 Only** → Filter warning alerts
- **Today** → Show today's incidents only
- **Export to CSV** → Download logs as spreadsheet
- **Clear All Logs** → Delete all log entries (requires confirmation)

## 🚨 Alert Levels

### Level 1 - Unsafe Movement (⚠️ Orange)
- **Confidence**: 50-64%
- **Meaning**: Potential concern, monitoring recommended
- **Response**: Observe swimmer, prepare for intervention
- **Notification**: Dismissible alert

### Level 2 - Drowning Emergency (🚨 Red)
- **Confidence**: 65%+ OR 80%+ OR 3+ seconds
- **Meaning**: High probability of drowning
- **Response**: Immediate intervention required
- **Notification**: Persistent alert (requires interaction)

## 🎯 Detection Triggers

| Condition | Confidence | Duration | Alert Level |
|-----------|-----------|----------|-------------|
| Monitoring | 50-64% | Any | Level 1 |
| Erratic Movement | 65-79% | Any | Level 2 |
| Critical | 80%+ | Instant | Level 2 |
| Submersion | 50%+ | 3+ seconds | Level 2 |

## 📊 Performance Settings

### Quick Adjustments in `performance_settings.py`:

**For Smooth Webcam:**
```python
PROCESS_EVERY_N_FRAMES = 1
SCALE_FACTOR = 0.6
JPEG_QUALITY = 75
```

**For Maximum Speed:**
```python
PROCESS_EVERY_N_FRAMES = 3
SCALE_FACTOR = 0.5
JPEG_QUALITY = 60
```

**For Best Quality:**
```python
PROCESS_EVERY_N_FRAMES = 1
SCALE_FACTOR = 0.75
JPEG_QUALITY = 85
```

## 🔧 Common Commands

### Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Update packages
pip install --upgrade -r requirements.txt
```

### Server Management
```bash
# Start server
python app.py

# Start with specific port
# Edit app.py: app.run(port=8080)

# Stop server
Ctrl + C
```

### Network Access
```bash
# Find your IP (Windows)
ipconfig

# Find your IP (macOS/Linux)
ifconfig

# Access from other device
http://YOUR-IP:5000
```

## 🧪 Testing Functions

Open browser console (F12) and run:

```javascript
// Test Level 1 warning
testLevel1()

// Test Level 2 emergency
testLevel2()

// Check detection polling status
console.log(detectionPollingInterval)

// Manually trigger detection check
checkForDetections()
```

## 📁 File Structure

```
Final YOLOv11 Dashboard/
├── app.py                    # Main Flask server
├── performance_settings.py   # Performance config
├── requirements.txt          # Python dependencies
├── best.pt                   # Trained model (96.92% accuracy)
├── README.md                 # Full documentation
├── SETUP_GUIDE.md           # Installation guide
├── QUICK_REFERENCE.md       # This file
├── .gitignore               # Git ignore rules
├── templates/
│   └── dashboard_live.html  # Web interface
├── uploads/                 # Video uploads (auto-created)
└── scripts/
    └── start_dashboard.ps1  # Quick start (Windows)
```

## 🔍 Troubleshooting Quick Fixes

### "Module not found"
```bash
pip install --upgrade -r requirements.txt
```

### "Port already in use"
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9
```

### "Camera not working"
1. Close other camera apps
2. Refresh browser (Ctrl + F5)
3. Allow camera permissions
4. Try different browser

### "Low FPS"
Edit `performance_settings.py`:
```python
PROCESS_EVERY_N_FRAMES = 3
SCALE_FACTOR = 0.5
```

### "Level 2 not triggering"
1. Lower confidence threshold to 50%
2. Check 3-second duration setting
3. Verify detection in console logs

## 📋 Incident Log Management

### View Logs
1. Switch to "📋 Incident Logs" tab
2. See all alerts with timestamp, level, confidence

### Filter Logs
- Click filter buttons: All, Level 2, Level 1, Today

### Edit Notes
1. Click ✏️ Edit on any log entry
2. Type your notes
3. Click 💾 Save

### Delete Log
1. Click 🗑️ Delete on log entry
2. Confirm deletion
3. Log is permanently removed

### Export Logs
1. Click 💾 Export to CSV
2. File downloads: `pool-alert-logs-YYYY-MM-DD.csv`
3. Open in Excel or Google Sheets

### Clear All Logs
1. Click 🗑️ Clear All Logs
2. Confirm action
3. All logs deleted (cannot be undone)

## 🎛️ Keyboard Shortcuts

Browser Console (F12):
- `testLevel1()` - Create test Level 1 alert
- `testLevel2()` - Create test Level 2 emergency
- `switchTab('detection')` - Go to Detection tab
- `switchTab('logs')` - Go to Logs tab
- `exportLogs()` - Export logs to CSV
- `clearAllLogs()` - Clear all logs (with confirmation)

## 📊 Statistics Cards

Dashboard shows real-time statistics:

1. **🚨 Level 2 Emergencies** - Count of drowning alerts
2. **⚠️ Level 1 Warnings** - Count of unsafe movement
3. **📅 Today's Incidents** - All alerts today
4. **📊 Total Alerts** - Lifetime count

## 🔔 Browser Notifications

### Enable Notifications
- Browser will request permission on first load
- Click "Allow" to enable

### Notification Types
- **Level 1**: Dismissible notification (orange)
- **Level 2**: Persistent notification (red, requires interaction)

### Disable Notifications
- Browser settings → Notifications → Block localhost:5000

## ⚙️ Advanced Configuration

### Change Level 2 Duration Threshold
Edit `app.py`:
```python
LEVEL_2_DURATION_THRESHOLD = 3.0  # Change to 5.0 for 5 seconds
```

### Change Confidence Thresholds
Edit `app.py` detection logic:
```python
if conf_percentage >= 80:   # Critical
if conf_percentage >= 65:   # Erratic
else:                       # Level 1 (50-64%)
```

### Change Max Upload Size
Edit `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
```

## 📞 Support Resources

1. **README.md** - Complete documentation
2. **SETUP_GUIDE.md** - Installation instructions
3. **Console logs** - Browser F12 → Console tab
4. **Server logs** - Terminal output from `python app.py`

## 🎯 Best Practices

1. ✅ Test system before deployment
2. ✅ Train staff on alert response
3. ✅ Review logs regularly
4. ✅ Export logs for record-keeping
5. ✅ Adjust confidence based on false positive rate
6. ✅ Use GPU for better performance (if available)
7. ✅ Monitor system resources (CPU, RAM)
8. ✅ Keep model file (best.pt) backed up
9. ✅ Update dependencies periodically
10. ✅ Never rely solely on AI - maintain human supervision

## 🚨 Emergency Response Protocol

### When Level 2 Alert Triggers:

1. **Immediate**: Look at video feed to verify
2. **Visual Confirmation**: Check actual pool situation
3. **Response**: Deploy rescue personnel if confirmed
4. **Log**: Add notes to incident log with outcome
5. **Review**: Analyze detection after incident

### When Level 1 Alert Triggers:

1. **Observe**: Monitor swimmer closely
2. **Prepare**: Ready rescue equipment
3. **Assess**: Determine if intervention needed
4. **Log**: Document observation and action taken

---

**Last Updated**: October 2025
**Version**: 1.0.0
**Model**: YOLOv11n (96.92% accuracy)
