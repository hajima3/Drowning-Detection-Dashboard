# 🔗 Multi-Model Ensemble Guide

## Quick Start: Combining Multiple Models

### Step 1: Prepare Your Models Folder

```powershell
# Create models directory
mkdir models
```

### Step 2: Add Your Models

Place your trained models in the project:
```
yolov11dashboard/
├── best.pt                    # Your original model
├── models/
│   ├── roboflow_model.pt     # From Roboflow
│   ├── best_v2.pt            # Another trained version
│   └── custom_pool.pt        # Custom trained model
```

### Step 3: Configure Ensemble

Edit `model_config.py`:

```python
# Enable ensemble
ENABLE_ENSEMBLE = True

# Add your models
ENSEMBLE_MODELS = [
    'best.pt',
    'models/roboflow_model.pt',
    'models/best_v2.pt',
]

# Choose strategy
ENSEMBLE_STRATEGY = 'average'  # Recommended for similar models
```

### Step 4: Run Dashboard

```powershell
python app.py
```

---

## 🎯 Ensemble Strategies Explained

### 1. **Average** (Recommended)
```python
ENSEMBLE_STRATEGY = 'average'
```
- Averages confidence scores from all models
- Best for: Similar models trained on related datasets
- Example: Combining Roboflow + your custom model

### 2. **Max Confidence**
```python
ENSEMBLE_STRATEGY = 'max'
```
- Takes prediction with highest confidence
- Best for: Diverse models with different strengths
- Example: General drowning model + specific pool model

### 3. **Voting**
```python
ENSEMBLE_STRATEGY = 'vote'
MIN_MODELS_AGREEMENT = 2
```
- Requires multiple models to agree
- Best for: Reducing false positives
- Example: Need 2+ models to detect before alerting

### 4. **Weighted**
```python
ENSEMBLE_STRATEGY = 'weighted'
ENSEMBLE_WEIGHTS = [0.5, 0.3, 0.2]  # 50%, 30%, 20%
```
- Trusts certain models more
- Best for: When one model is clearly better
- Example: 50% Roboflow, 30% custom, 20% backup

---

## 📊 Real-World Example

### Scenario: Roboflow + 2 Custom Models

```python
# model_config.py
ENABLE_ENSEMBLE = True

ENSEMBLE_MODELS = [
    'best.pt',                      # Your original (indoor pools)
    'models/roboflow_outdoor.pt',   # Roboflow (outdoor pools)
    'models/custom_night.pt',       # Night vision trained
]

ENSEMBLE_STRATEGY = 'weighted'
ENSEMBLE_WEIGHTS = [0.4, 0.4, 0.2]  # Equal trust in first two

MIN_MODELS_AGREEMENT = 2  # Need 2/3 to agree
```

**Benefits:**
- ✅ Better detection in various lighting conditions
- ✅ Works for indoor AND outdoor pools
- ✅ Reduced false alarms (2/3 consensus)

---

## 🚀 Adding Roboflow Model

### From Roboflow:

1. **Train** your model in Roboflow
2. **Export** as: `YOLOv11 PyTorch`
3. **Download** the `best.pt` file
4. **Rename** to `roboflow_model.pt`
5. **Move** to `models/` folder

### Configure:

```python
ENSEMBLE_MODELS = [
    'best.pt',                    # Original
    'models/roboflow_model.pt',   # New Roboflow model
]
```

---

## ⚙️ Advanced Settings

### Confidence Adjustments

Boost or reduce confidence for specific models:

```python
MODEL_CONFIDENCE_ADJUSTMENTS = {
    0: 1.1,   # Boost first model by 10%
    1: 0.9,   # Reduce second model by 10%
    2: 1.0,   # Keep third model unchanged
}
```

### Stricter Consensus

Require more models to agree:

```python
MIN_MODELS_AGREEMENT = 3  # Need 3+ models (out of 4) to detect
```

### Custom NMS Threshold

Control overlapping box removal:

```python
ENSEMBLE_NMS_THRESHOLD = 0.3  # More strict (removes more overlaps)
```

---

## 🎮 Testing Your Ensemble

### Test with Single Model (Baseline)

```python
ENABLE_ENSEMBLE = False
```

Run dashboard, note accuracy/false positives.

### Enable Ensemble

```python
ENABLE_ENSEMBLE = True
ENSEMBLE_MODELS = ['best.pt', 'models/roboflow_model.pt']
ENSEMBLE_STRATEGY = 'average'
```

Run dashboard again, compare results!

### Compare Strategies

Try each strategy with same videos:
- Average → Best balance
- Max → Most detections
- Vote → Fewest false positives
- Weighted → Custom tuning

---

## 📈 Performance Considerations

### Speed vs Accuracy

| Models | FPS Impact | Accuracy Boost |
|--------|-----------|----------------|
| 1 model | 30 FPS | Baseline |
| 2 models | ~20 FPS | +5-10% |
| 3 models | ~15 FPS | +8-15% |
| 4+ models | <10 FPS | Diminishing returns |

**Recommendation:** 2-3 models for best balance

### Optimize Performance

If ensemble is too slow:

```python
# In performance_settings.py
PROCESS_EVERY_N_FRAMES = 3  # Skip more frames
SCALE_FACTOR = 0.6          # Lower resolution
```

---

## 🔧 Troubleshooting

### Models Not Loading

```
❌ Failed to load models/roboflow_model.pt
```

**Fix:** Check file path and permissions:
```powershell
ls models/  # Verify file exists
```

### Ensemble Too Slow

**Fix 1:** Reduce models
```python
ENSEMBLE_MODELS = ['best.pt', 'models/roboflow_model.pt']  # Use only 2
```

**Fix 2:** Adjust frame processing
```python
PROCESS_EVERY_N_FRAMES = 4  # Process every 4th frame
```

### Different Class Names

If models have different classes (drowning/swimming):

**Current support:** Assumes class 0 = drowning, class 1 = swimming

To customize, edit class mapping in `app.py` > `combine_by_vote()` function.

---

## 💡 Best Practices

1. **Train Similar Models**
   - Same resolution
   - Similar training data
   - Same class labels

2. **Start with 2 Models**
   - Test baseline performance
   - Add more only if needed

3. **Use Weighted Strategy**
   - When one model is clearly superior
   - Give it higher weight (0.6+)

4. **Enable Voting for Critical Applications**
   - Reduces false alarms
   - Better for public pools

5. **Test Extensively**
   - Try different strategies
   - Compare results
   - Find best balance

---

## 📞 Need Help?

If you have issues:

1. Check `model_config.py` syntax
2. Verify all model files exist
3. Test single model first
4. Check terminal for error messages

---

**Next Steps:**
1. ✅ Create `models/` folder
2. ✅ Add your Roboflow model
3. ✅ Update `model_config.py`
4. ✅ Run and test!
