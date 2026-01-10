"""
Multi-Model Configuration for YOLOv11 Ensemble
Place your multiple best.pt models in the models/ folder
Rename them like: best_v1.pt, best_v2.pt, roboflow_model.pt, etc.
"""

# ============================================
# MODEL ENSEMBLE SETTINGS
# ============================================

# Set to True to enable multi-model ensemble
ENABLE_ENSEMBLE = False

# List of model paths (relative to project root)
# Example: ['best.pt', 'models/roboflow_model.pt', 'models/custom_model.pt']
ENSEMBLE_MODELS = [
    'best.pt',
    # 'models/best_v2.pt',
    # 'models/roboflow_model.pt',
]

# Ensemble strategy:
# 'vote' - Majority voting (requires 2+ models to agree)
# 'average' - Average confidence scores
# 'max' - Take highest confidence prediction
# 'weighted' - Weighted average (use ENSEMBLE_WEIGHTS)
ENSEMBLE_STRATEGY = 'average'

# Weights for each model (only used if strategy='weighted')
# Must match the number of models in ENSEMBLE_MODELS
ENSEMBLE_WEIGHTS = [1.0]  # Example: [0.5, 0.3, 0.2] for 3 models

# Minimum confidence threshold for ensemble predictions
ENSEMBLE_MIN_CONFIDENCE = 0.4

# ============================================
# ADVANCED SETTINGS
# ============================================

# NMS (Non-Maximum Suppression) IoU threshold for combining boxes
# Lower = more strict (removes more overlapping boxes)
ENSEMBLE_NMS_THRESHOLD = 0.5

# Require at least N models to detect before accepting prediction
MIN_MODELS_AGREEMENT = 1  # Set to 2+ for stricter consensus

# Enable model-specific confidence boost/penalty
# Format: {model_index: boost_factor}
# Example: {0: 1.1, 1: 0.9} boosts first model by 10%, reduces second by 10%
MODEL_CONFIDENCE_ADJUSTMENTS = {}

# ============================================
# USAGE INSTRUCTIONS
# ============================================
"""
HOW TO ADD MULTIPLE MODELS:

1. Create a 'models' folder in your project root:
   mkdir models

2. Add your trained models:
   - best.pt (your current model)
   - models/roboflow_model.pt (from Roboflow)
   - models/custom_model_v2.pt (another version)

3. Update ENSEMBLE_MODELS list:
   ENSEMBLE_MODELS = [
       'best.pt',
       'models/roboflow_model.pt',
       'models/custom_model_v2.pt',
   ]

4. Enable ensemble:
   ENABLE_ENSEMBLE = True

5. Choose strategy:
   - 'average' - Best for similar models (RECOMMENDED)
   - 'max' - Best for diverse models, takes highest confidence
   - 'vote' - Best for consensus (requires 2+ models to agree)
   - 'weighted' - Best when you trust one model more

6. (Optional) Set weights for weighted strategy:
   ENSEMBLE_WEIGHTS = [0.5, 0.3, 0.2]  # 50%, 30%, 20%

BENEFITS:
✅ Higher accuracy (combined predictions)
✅ Reduced false positives
✅ More robust detection
✅ Better generalization

CONSIDERATIONS:
⚠️  Slower inference (processes multiple models)
⚠️  Higher memory usage
⚠️  Recommended: 2-4 models (more isn't always better)
"""
