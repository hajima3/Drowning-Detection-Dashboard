# Models Folder

Place your additional trained models here.

## Quick Guide

### Adding a Roboflow Model:

1. Train model in Roboflow
2. Export as **YOLOv11 PyTorch**
3. Download `best.pt`
4. Rename to `roboflow_model.pt`
5. Place in this folder

### File Structure:

```
models/
├── README.md (this file)
├── roboflow_model.pt     # From Roboflow
├── best_v2.pt            # Another version
└── custom_model.pt       # Custom trained
```

### Enable in model_config.py:

```python
ENABLE_ENSEMBLE = True
ENSEMBLE_MODELS = [
    'best.pt',
    'models/roboflow_model.pt',
    'models/best_v2.pt',
]
```

See `ENSEMBLE_GUIDE.md` for complete instructions!
