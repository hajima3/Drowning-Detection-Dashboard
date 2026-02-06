# 📊 5-Iteration Training Plan

## Overview
Track model accuracy improvements across 5 training iterations from Label Studio.

---

## Iteration Workflow

### **Before Each Iteration:**
1. Collect & annotate data in Label Studio
2. Train YOLOv11 model
3. Evaluate metrics (Precision, Recall, F1, mAP50)
4. Log results using model_manager.py
5. Deploy if F1 ≥ 95%

---

## Iteration Plan

### 🔄 **Iteration 1: Baseline Model**
**Goal:** Establish baseline performance  
**Dataset Target:** 500-800 samples  

**Steps:**
```bash
# 1. Train in Label Studio
# 2. After training, log results:
python model_manager.py add 1 <precision> <recall> <f1> <map50> <dataset_size> "Baseline model"

# Example:
python model_manager.py add 1 88.5 90.2 89.3 89.8 600 "Initial baseline - needs improvement"
```

**Expected Results:**
- F1 Score: 85-92% (baseline)
- Status: needs_improvement

---

### 🔄 **Iteration 2: Data Expansion**
**Goal:** Add diverse drowning scenarios  
**Dataset Target:** 1000-1500 samples (500-700 new)

**Focus Areas:**
- Different pool types (indoor/outdoor)
- Various lighting conditions
- Different age groups
- More drowning behavior examples

**Steps:**
```bash
# After training:
python model_manager.py add 2 <precision> <recall> <f1> <map50> <dataset_size> "Added diverse scenarios"

# Example:
python model_manager.py add 2 92.3 93.1 92.7 93.0 1200 "Improved with diverse data"
```

**Expected Results:**
- F1 Score: 90-94%
- Status: needs_improvement or approaching target

---

### 🔄 **Iteration 3: Balance & Refinement**
**Goal:** Balance classes, remove ambiguous samples  
**Dataset Target:** 1500-2000 samples (500 new)

**Focus Areas:**
- Equal drowning/swimming samples
- Remove unclear annotations
- Add edge cases
- Better annotation quality

**Steps:**
```bash
python model_manager.py add 3 <precision> <recall> <f1> <map50> <dataset_size> "Balanced dataset"

# Example:
python model_manager.py add 3 94.8 95.2 95.0 95.3 1800 "Balanced classes - reached target!"
```

**Expected Results:**
- F1 Score: 93-96%
- Status: **deployed** if F1 ≥ 95%

---

### ✅ **Iteration 4: Fine-tuning (if needed)**
**Goal:** Optimize for production edge cases  
**Dataset Target:** 2000-2500 samples (500 new)

**Focus Areas:**
- Difficult/rare drowning scenarios
- Reduce false positives
- Night/low-light conditions
- Crowded pool scenarios

**Steps:**
```bash
python model_manager.py add 4 <precision> <recall> <f1> <map50> <dataset_size> "Production optimization"

# Example:
python model_manager.py add 4 96.2 96.8 96.5 96.7 2200 "Optimized for edge cases"
```

**Expected Results:**
- F1 Score: 95-97%
- Status: **deployed**

---

### 🏆 **Iteration 5: Final Production Model**
**Goal:** Maximum accuracy for deployment  
**Dataset Target:** 2500-3000 samples (500 new)

**Focus Areas:**
- Real-world pool footage
- Production environment testing
- Stress testing edge cases
- Final optimization

**Steps:**
```bash
python model_manager.py add 5 <precision> <recall> <f1> <map50> <dataset_size> "Final production model"

# Example:
python model_manager.py add 5 97.5 97.8 97.6 97.9 2800 "Production-ready - final model"
```

**Expected Results:**
- F1 Score: 96-98%
- Status: **deployed**
- Ready for real pools!

---

## Commands Reference

### View All Iterations
```bash
python model_manager.py summary
```

### Add New Iteration
```bash
python model_manager.py add <iteration> <precision> <recall> <f1> <map50> <dataset_size> "notes"
```

### Deploy Model After Training
```bash
# 1. Copy trained model from Label Studio
cp path/to/best.pt best_v<iteration>.pt

# 2. If F1 ≥ 95%, deploy to dashboard
cp best_v<iteration>.pt best.pt

# 3. Restart dashboard
python app.py
```

---

## Tracking Progress

### Check Dashboard
1. Open: http://localhost:5000
2. Click: **📊 Model Metrics** tab
3. View: Iteration history and F1 progression

### Expected F1 Score Progression
```
Iteration 1: 85-92%  ❌ (baseline)
Iteration 2: 90-94%  🔄 (improving)
Iteration 3: 93-96%  ✅ (target reached!)
Iteration 4: 95-97%  ✅ (optimized)
Iteration 5: 96-98%  🏆 (production ready)
```

---

## Success Criteria

### Minimum Requirements (Iteration 3+)
- ✅ **Precision:** ≥ 95%
- ✅ **Recall:** ≥ 95%
- ✅ **F1 Score:** ≥ 95%
- ✅ **mAP50:** ≥ 95%

### Production Ready (Iteration 4-5)
- ✅ **F1 Score:** 96-98%
- ✅ Tested on real pool footage
- ✅ False positive rate < 5%
- ✅ False negative rate < 3%

---

## Example Complete Workflow

```bash
# Iteration 1: Initial Training
python model_manager.py add 1 88.5 90.2 89.3 89.8 600 "Baseline model"

# Iteration 2: More data
python model_manager.py add 2 92.3 93.1 92.7 93.0 1200 "Added diverse scenarios"

# Iteration 3: Balanced dataset - REACHED TARGET!
python model_manager.py add 3 95.2 95.8 95.5 95.7 1800 "Balanced dataset - deployed"
cp best_v3.pt best.pt  # Deploy to dashboard

# Iteration 4: Optimization
python model_manager.py add 4 96.5 96.9 96.7 96.8 2200 "Edge case optimization"
cp best_v4.pt best.pt  # Update dashboard

# Iteration 5: Final model
python model_manager.py add 5 97.5 97.8 97.6 97.9 2800 "Production model"
cp best_v5.pt best.pt  # Final deployment

# View results
python model_manager.py summary
```

---

## Tips for Success

1. **Don't rush iterations** - Quality > quantity
2. **Test thoroughly** between iterations
3. **Keep all model versions** - `best_v1.pt`, `best_v2.pt`, etc.
4. **Document changes** - Use detailed notes
5. **Balance your dataset** - Equal drowning/swimming samples
6. **Verify in dashboard** - Test before production deployment

---

## Stopping Early

If you reach 95%+ F1 score before Iteration 5:
- ✅ You can stop and deploy!
- Continue only for optimization
- Focus on edge cases and production testing

**Remember:** 95% F1 is production-ready. 97-98% is excellent!
