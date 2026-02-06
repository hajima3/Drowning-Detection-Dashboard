# Model Deployment Guide

## Label Studio Training Workflow

### 1. Annotation in Label Studio
- Upload your drowning/swimming videos
- Annotate frames with bounding boxes:
  - Class 0: **Drowning** (person in distress, submerged, erratic movement)
  - Class 1: **Swimming** (normal swimming behavior)
- Export annotated dataset

### 2. Model Training
Train YOLOv11 with your annotated data:

```bash
yolo train data=your_dataset.yaml model=yolov11n.pt epochs=100 imgsz=640
```

### 3. Model Evaluation
Check your training results:

**Required Metrics:**
- **Precision**: 95-98% (minimize false alarms)
- **Recall**: 95-98% (catch all drowning cases)
- **F1 Score**: 95-98% (final accuracy metric)
- **mAP50**: 95%+ (mean average precision at IoU 0.5)

**Formula:**
```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example Good Results:**
```
Precision: 96.5%
Recall: 97.2%
F1 Score: 96.8% ✅ (meets 95-98% target)
mAP50: 96.9%
```

**Example Needs More Training:**
```
Precision: 88.3%
Recall: 91.5%
F1 Score: 89.9% ❌ (below 95% target - retrain with more data)
```

### 4. Deploy to Dashboard

Once your model meets the 95-98% F1 score target:

1. **Find your trained model:**
   - Located in `runs/detect/train/weights/best.pt`

2. **Copy to dashboard:**
   ```bash
   cp runs/detect/train/weights/best.pt c:/Users/vince/OneDrive/Desktop/yolov11dashboard/best.pt
   ```

3. **Restart dashboard:**
   ```bash
   cd c:/Users/vince/OneDrive/Desktop/yolov11dashboard
   python app.py
   ```

4. **Test in production:**
   - Start webcam detection
   - Verify confidence scores match your training metrics
   - Adjust confidence threshold if needed (default: 50%)

### 5. Confidence Threshold Guidelines

Based on your F1 score:

- **F1: 95-96%** → Set threshold: 45-50%
- **F1: 96-97%** → Set threshold: 50-55%
- **F1: 97-98%** → Set threshold: 55-60%

Higher F1 scores allow higher thresholds with fewer false positives.

## Retraining Tips

If your model is below 95% F1:

1. **Add more diverse data:**
   - Different pool environments (indoor/outdoor)
   - Various lighting conditions
   - Different age groups and body types
   - More drowning behavior examples

2. **Balance your dataset:**
   - Equal drowning and swimming examples
   - Minimum 500+ images per class

3. **Train longer:**
   - Increase epochs (100-200)
   - Use data augmentation

4. **Try different YOLOv11 models:**
   - YOLOv11n (nano - fast, less accurate)
   - YOLOv11s (small - balanced)
   - YOLOv11m (medium - more accurate)

## Production Deployment Checklist

Before deploying to real pools:

- ✅ F1 Score: 95-98%
- ✅ Precision: 95%+ (few false alarms)
- ✅ Recall: 95%+ (catches all drownings)
- ✅ Tested on diverse video samples
- ✅ Audio alerts configured
- ✅ Dashboard tested with live webcam
- ✅ Staff trained on alert levels

## Model Version Tracking

Keep track of your models:

```
best_v1.pt - F1: 93.5% ❌ Not ready
best_v2.pt - F1: 95.8% ✅ Production ready
best_v3.pt - F1: 97.2% ✅ Current production
```

Always backup your current production model before replacing!
