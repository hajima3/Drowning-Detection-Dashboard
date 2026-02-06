# 🏊 Pool Behavior Classification System

## Overview
Comprehensive 3-level classification system for pool safety monitoring with 60+ behavior classes.

---

## 📊 Alert Level Structure

### **Level 0 - Normal Behavior** ✅
Safe swimming and proper pool activities. No intervention needed.

**Categories:**
1. **Advanced/Coordinated Swimming**
   - Side stroke (controlled, symmetrical)
   - Butterfly (face forward/down, arms wide)
   - Underwater swimming (short period, calm)

2. **Vertical/Survival Swimming**
   - Treading water (calm, stable)
   - Dog paddle (vertical, head above water)

3. **Horizontal Stroke Swimming**
   - Freestyle (facing down, alternating arms)
   - Backstroke (face up, horizontal)
   - Breaststroke (synchronized movements)

4. **Resting Behavior**
   - Back float (horizontal, face above)
   - Treading water (minimal movement)
   - Vertical rest (upright, calm)

5. **Pool Entry Behavior**
   - Diving (head-first, controlled)
   - Jumping (feet-first, controlled)
   - Walking in (minimal splash)

6. **Following Rules**
   - Proper swim wear (rash guards, swim trunks, caps, goggles, one-piece)
   - Appropriate movements (sitting, walking, standing, stretching)

**Total Classes:** 19

---

### **Level 1 - Unsafe/Erratic** ⚠️
Potentially dangerous behavior requiring staff monitoring and intervention.

**Categories:**
1. **Rule Violations**
   - Not following pool rules
   - Improper movement
   - Running on deck
   - Attempting dangerous stunts
   - Rough housing

2. **Improper Attire**
   - Heavy street clothes
   - Loose accessories
   - Water-absorbent clothes

3. **Unsafe Entry**
   - Unsafe diving (wrong angle, looking backward)
   - Improper pool entry
   - Stumble entry
   - Body hits water hard
   - Entering non-designated areas

4. **Erratic Pool Movement**
   - Wall clutching with flailing
   - Spinning in circles
   - Unstable floating
   - Continuous sinking
   - Head repeatedly going underwater

5. **Swimming Difficulties**
   - Poor breath control
   - Loss of body balance
   - Panicking/fatigue signs
   - Panicked expression
   - Erratic swimming patterns

6. **Improper Swimming Techniques**
   - Bobbing head (repeatedly sinks)
   - Broken stroke rhythm
   - Body drags deep in water
   - Panic lifting to breathe
   - Uneven arm reach
   - Excessive body sway
   - High-splash kick

**Total Classes:** 42

---

### **Level 2 - Critical Drowning** 🚨
Life-threatening situations requiring immediate emergency response.

**Categories:**
1. **Active Drowning**
   - Active sinking
   - Descending depth rapidly
   - Up and down movement (vertical struggle)
   - Head underwater for extended duration

2. **Distress Signals**
   - Lack of movement
   - Erratic splashing
   - High splashing (panic)
   - Waving arms
   - Panicked movements
   - Only limbs visible (body submerged)

3. **Submersion Indicators**
   - Head fully submerged
   - Panic treading (uncontrolled)
   - Face-down position
   - Body moves only with pool current

4. **Physical Collapse**
   - Physical collapse in water
   - Limp limbs (unconscious)
   - Claw-like hands (instinctive drowning response)
   - Slowly sinking (passive drowning)

5. **Critical Position**
   - Vertical body (unable to horizontal)
   - Distress signs visible
   - Uncontrolled arm/leg movement

**Total Classes:** 21

---

## 🔄 Integration with Label Studio

### Annotation Guidelines

When annotating in Label Studio, use these **exact class names** (use underscores):

#### Level 0 Examples:
```
controlled_swimming
side_stroke
butterfly
freestyle
backstroke
breaststroke
treading_water
dog_paddle
diving
jumping
back_float
```

#### Level 1 Examples:
```
running
rough_housing
improper_swim_wear
unsafe_diving
wall_clutching
flailing_arms
erratic_pool_movement
bobbing_head
poor_breath_control
panicking
fatigue_movements
```

#### Level 2 Examples:
```
active_sinking
descending_depth
head_underwater_long
panic_treading
face_down_position
physical_collapse
limp_limbs
vertical_body
claw_like_hands
slowly_sinking
```

---

## 📝 How It Works in Dashboard

### Detection Flow:
1. **YOLOv11 detects** behavior class in video frame
2. **Dashboard looks up** class in `class_mapping.json`
3. **Alert level assigned** automatically (Level 0, 1, or 2)
4. **Alert triggered** based on level:
   - Level 0: No alert (normal swimming)
   - Level 1: Orange warning + level1.mp3 audio
   - Level 2: Red emergency + level2.mp3 audio
5. **Incident logged** with class name and confidence

### Example Detection:
```
Detected: "panic_treading" at 87% confidence
→ Mapped to: Level 2 - Critical Drowning
→ Alert: 🚨 LEVEL 2 EMERGENCY: panic_treading
→ Audio: level2.mp3 plays
→ Log: Saved with timestamp and class
```

---

## 🎯 Training Priority by Level

### Phase 1: Level 2 Classes (CRITICAL)
Focus first on life-threatening behaviors:
- active_sinking
- face_down_position
- panic_treading
- limp_limbs
- head_underwater_long

**Goal:** Minimize false negatives (never miss a drowning)

### Phase 2: Level 1 Classes (WARNING)
Add unsafe behaviors:
- running
- rough_housing
- wall_clutching
- bobbing_head
- unsafe_diving

**Goal:** Early intervention before escalation

### Phase 3: Level 0 Classes (NORMAL)
Complete with normal behaviors:
- All swimming strokes
- Proper pool entry
- Resting behaviors

**Goal:** Reduce false positives

---

## 📊 Expected Iteration Progress

### Iteration 1 (Baseline)
- **Classes:** 5-10 critical Level 2 behaviors
- **Dataset:** 500-800 samples
- **F1 Target:** 85-90%

### Iteration 2 (Expansion)
- **Classes:** Add 10-15 Level 1 behaviors
- **Dataset:** 1200-1500 samples
- **F1 Target:** 90-93%

### Iteration 3 (Balanced)
- **Classes:** Add 10-15 Level 0 behaviors
- **Dataset:** 1800-2200 samples
- **F1 Target:** 93-96% ✅

### Iteration 4 (Refinement)
- **Classes:** All 60+ behaviors
- **Dataset:** 2200-2600 samples
- **F1 Target:** 95-97%

### Iteration 5 (Production)
- **Classes:** Fine-tune all 60+ behaviors
- **Dataset:** 2800-3200 samples
- **F1 Target:** 96-98% 🏆

---

## ⚙️ Configuration Files

### `class_mapping.json`
Defines which classes belong to which alert level. The dashboard reads this file to automatically assign alerts.

**Structure:**
```json
{
  "alert_levels": {
    "level_0": {
      "name": "Normal Behavior",
      "classes": ["freestyle", "backstroke", ...]
    },
    "level_1": {
      "name": "Unsafe / Erratic Movement",
      "classes": ["running", "rough_housing", ...]
    },
    "level_2": {
      "name": "Critical Drowning Emergency",
      "classes": ["active_sinking", "panic_treading", ...]
    }
  }
}
```

### Adding New Classes
1. Open `class_mapping.json`
2. Add class name to appropriate level array
3. Use underscores: `"new_behavior_name"`
4. Restart dashboard

---

## 🎓 Training Tips

### Multi-Class Training
- Start with **high-priority classes** (Level 2)
- Add classes **gradually** across iterations
- Ensure **balanced samples** per class
- Use **clear, distinct** class names
- Maintain **annotation consistency**

### Class Naming Convention
- Use lowercase
- Separate words with underscores
- Be specific: `panic_treading` not just `treading`
- Match exactly with `class_mapping.json`

### Sample Distribution
**Recommended per iteration:**
- Level 2: 40% of samples (highest priority)
- Level 1: 35% of samples
- Level 0: 25% of samples

**Example Iteration 3:**
- 800 Level 2 samples (active_sinking, panic_treading, etc.)
- 650 Level 1 samples (running, rough_housing, etc.)
- 550 Level 0 samples (freestyle, backstroke, etc.)
- **Total:** 2000 samples

---

## 🚀 Quick Start

1. **Train in Label Studio** with exact class names
2. **Export model** as `best.pt`
3. **Deploy to dashboard** - automatic class mapping!
4. **Test detection** - verify classes trigger correct alerts
5. **Log iteration** with model_manager.py

**The dashboard handles everything automatically based on your class names!**
