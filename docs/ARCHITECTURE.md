# 🏗️ Project Architecture

## Overview
This YOLOv11 drowning detection system follows a modular architecture designed for easy maintenance, testing, and future feature integration.

## Directory Structure

```
yolov11dashboard/
│
├── src/                          # Source code modules
│   ├── core/                     # Core utilities
│   │   ├── __init__.py
│   │   └── config_loader.py     # Configuration management
│   │
│   ├── inference/                # AI inference module
│   │   ├── __init__.py
│   │   └── model_inference.py   # YOLOv11 model operations
│   │
│   ├── alerts/                   # Alert management
│   │   ├── __init__.py
│   │   └── alert_manager.py     # Alert levels & notifications
│   │
│   └── __init__.py
│
├── config/                       # Configuration files
│   ├── config.yaml              # Main configuration
│   └── .env.template            # Environment variables template
│
├── models/                       # YOLOv11 model files
│   ├── best.pt                  # Primary trained model
│   └── README.md                # Model management guide
│
├── datasets/                     # Training/validation data
│   ├── local/                   # Local datasets
│   │   ├── drowning/
│   │   └── swimming/
│   └── internet/                # Downloaded datasets
│
├── templates/                    # Flask HTML templates
│   └── dashboard_live.html      # Web dashboard UI
│
├── uploads/                      # Video file uploads
│
├── scripts/                      # Utility scripts
│   └── start_dashboard.ps1      # Quick start script
│
├── docs/                         # Consolidated documentation
│   └── (documentation files)
│
├── app.py                        # Flask application (to be refactored)
├── performance_settings.py       # Legacy performance config
├── model_config.py              # Legacy model config
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Main project documentation
```

## Module Responsibilities

### 1. `src/core/` - Core Utilities
**Purpose**: Central configuration and shared utilities

**Components**:
- `config_loader.py`: Loads and manages configuration from YAML/env files
  - Singleton pattern for global access
  - Environment variable override support
  - Type-safe configuration access

**Usage**:
```python
from src.core import get_config

config = get_config()
model_path = config.get('MODEL.PRIMARY_MODEL')
```

---

### 2. `src/inference/` - AI Inference
**Purpose**: YOLOv11 model loading and prediction

**Components**:
- `model_inference.py`: Model operations
  - Single model inference
  - Multi-model ensemble support
  - Strategy-based prediction combination (average, max, vote, weighted)

**Usage**:
```python
from src.inference import ModelInference

# Single model
inference = ModelInference('models/best.pt')
results = inference.predict(frame, conf_threshold=0.5)

# Ensemble
inference = ModelInference(
    ['models/best.pt', 'models/roboflow.pt'],
    enable_ensemble=True,
    ensemble_strategy='average'
)
results = inference.predict(frame)
```

---

### 3. `src/alerts/` - Alert Management
**Purpose**: Detection processing, alert level determination, and notifications

**Components**:
- `alert_manager.py`: Alert operations
  - Process detection results
  - Determine alert levels (Level 1, Level 2)
  - Track duration for escalation
  - Notification placeholders (SMS/Call)

**Features**:
- **Level 1**: 50-64% confidence → Warning
- **Level 2**: 65%+ confidence OR 3+ seconds → Emergency
- **Duration tracking**: Escalates persistent detections
- **Notification stubs**: Ready for Twilio/AWS SNS integration

**Usage**:
```python
from src.alerts import AlertManager

alert_mgr = AlertManager(alert_config, notification_config)
event = alert_mgr.process_detection(results, current_time)

if event:
    print(f"Alert Level {event.alert_level}: {event.confidence}%")
```

---

## Configuration Architecture

### config.yaml
Central configuration for all system components:

```yaml
MODEL:                    # Model paths & ensemble settings
ALERTS:                   # Alert thresholds & levels
NOTIFICATIONS:            # SMS/Call configuration (future)
PERFORMANCE:              # Processing optimizations
DATASETS:                 # Dataset paths & Roboflow integration
SERVER:                   # Flask server settings
LOGGING:                  # Log configuration
SECURITY:                 # API keys & SSL (future)
```

### .env (from .env.template)
Sensitive credentials and environment-specific values:
- Phone numbers
- API keys (Twilio, AWS, Roboflow)
- Database URLs
- Secret keys

**⚠️ Never commit .env to version control**

---

## Data Flow

### Detection Pipeline

```
1. Video Source (Webcam/File)
        ↓
2. Frame Capture & Preprocessing
        ↓
3. ModelInference.predict()
        ↓ (single or ensemble)
4. YOLO Detection Results
        ↓
5. AlertManager.process_detection()
        ↓
6. Alert Level Determination
        ↓
7. Notification Trigger (if enabled)
        ↓
8. Dashboard Update (Flask)
```

### Configuration Flow

```
1. config.yaml (defaults)
        ↓
2. .env overrides (secrets)
        ↓
3. config_loader.py (singleton)
        ↓
4. Used by: ModelInference, AlertManager, Flask app
```

---

## Future Integration Points

### 1. SMS/Call Notifications
**Location**: `src/alerts/alert_manager.py`

**Placeholders**:
- `_send_sms(event)` - Implement Twilio/AWS SNS
- `_initiate_call(event)` - Implement Twilio call API

**Config**: `config.yaml` → `NOTIFICATIONS` section

**Required**:
```bash
pip install twilio  # or boto3 for AWS SNS
```

### 2. Database Logging
**Location**: Create `src/database/` module

**Purpose**: Persistent storage for detection history

**Candidates**:
- SQLite (simple, local)
- PostgreSQL (production)
- MongoDB (flexible schema)

### 3. Roboflow Dataset Integration
**Location**: Create `src/datasets/roboflow_loader.py`

**Purpose**: Auto-download and sync datasets from Roboflow

**Config**: `config.yaml` → `DATASETS.ROBOFLOW`

### 4. API Authentication
**Location**: Create `src/core/auth.py`

**Purpose**: Secure Flask endpoints with API keys

**Config**: `config.yaml` → `SECURITY`

---

## Module Isolation Benefits

### ✅ Testability
Each module can be tested independently:
```python
# Test inference without Flask
inference = ModelInference('test_model.pt')
results = inference.predict(test_frame)
assert len(results.boxes) > 0
```

### ✅ Maintainability
Changes to one module don't affect others:
- Update alert thresholds → only `alert_manager.py`
- Change ensemble strategy → only `model_inference.py`
- Add SMS provider → only notification code

### ✅ Scalability
Easy to add new features:
- New notification channel → add to `alerts/`
- New model type → extend `inference/`
- New data source → add to `datasets/`

### ✅ Reusability
Modules can be used in other projects:
```python
# Use inference in a different app
from src.inference import ModelInference
```

---

## Configuration Management

### Loading Order
1. Read `config/config.yaml`
2. Override with `config/.env` (if exists)
3. Override with environment variables
4. Provide to modules via singleton

### Access Patterns
```python
# Global config
config = get_config()

# Specific sections
model_cfg = config.get_model_config()
alert_cfg = config.get_alert_config()

# Dot notation
value = config.get('MODEL.PRIMARY_MODEL')
```

---

## Migration Path

### Current State (app.py)
- Monolithic Flask application
- Inline model loading and detection
- Mixed concerns (routing + inference + alerts)

### Target State (Modular)
```python
# app.py (simplified)
from src.core import get_config
from src.inference import ModelInference
from src.alerts import AlertManager

config = get_config()
inference = ModelInference(...)
alert_mgr = AlertManager(...)

@app.route('/detect')
def detect():
    results = inference.predict(frame)
    event = alert_mgr.process_detection(results)
    return jsonify(event)
```

### Benefits After Refactor
- Clean separation of concerns
- Easy to test each component
- Simple to add features
- Configuration-driven behavior

---

## Development Workflow

### Adding a New Feature

1. **Update Config**
   - Add settings to `config/config.yaml`
   - Add secrets to `.env.template`

2. **Create/Update Module**
   - Add to appropriate `src/` directory
   - Follow existing patterns
   - Add `__init__.py` exports

3. **Integrate with Flask**
   - Import module in `app.py`
   - Add routes if needed
   - Update dashboard UI

4. **Test**
   - Unit test the module
   - Integration test with Flask
   - Manual test on dashboard

5. **Document**
   - Update this architecture doc
   - Add usage examples
   - Update README.md

---

## Best Practices

### ✅ DO
- Use configuration files for all settings
- Keep modules independent and focused
- Add type hints to function signatures
- Document complex logic
- Use environment variables for secrets

### ❌ DON'T
- Hard-code paths, thresholds, or credentials
- Mix routing logic with business logic
- Create circular dependencies between modules
- Commit `.env` files or API keys
- Make modules depend on Flask

---

## Quick Reference

### Import Paths
```python
from src.core import get_config
from src.inference import ModelInference
from src.alerts import AlertManager, DetectionEvent
```

### Configuration Access
```python
config = get_config()
config.get('MODEL.PRIMARY_MODEL')
config.get_alert_config()
```

### Model Inference
```python
inference = ModelInference('models/best.pt')
results = inference.predict(frame, conf_threshold=0.5)
```

### Alert Processing
```python
alert_mgr = AlertManager(alert_config, notif_config)
event = alert_mgr.process_detection(results, time.time())
```

---

This architecture provides a solid foundation for future enhancements while maintaining existing functionality.
