# Performance Optimization Settings for YOLOv11 Dashboard
# Adjust these values to balance speed vs quality
# Last updated: January 8, 2026

# ==================================================
# VIDEO PROCESSING SETTINGS
# ==================================================

# Frame Skip (Higher = Faster, but less smooth)
# - Set to 1: Process every frame (slowest, smoothest)
# - Set to 2: Process every other frame (balanced)
# - Set to 3: Process every 3rd frame (faster, may miss detections)
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame for better performance

# Resolution Scale (Lower = Faster)
# - Set to 1.0: Full resolution (slowest, best quality)
# - Set to 0.75: 75% size (balanced - RECOMMENDED)
# - Set to 0.5: Half size (fastest, lower quality)
SCALE_FACTOR = 0.5  # Reduced to 50% for faster processing

# JPEG Encoding Quality (Lower = Faster streaming)
# - Set to 95: High quality (slower streaming)
# - Set to 75: Medium quality (balanced - RECOMMENDED)
# - Set to 50: Lower quality (fastest streaming)
JPEG_QUALITY = 65  # Reduced for faster streaming

# ==================================================
# DETECTION SETTINGS
# ==================================================

# Default Confidence Threshold (0.1 to 0.9)
# - Lower (0.3-0.4): More detections, may have false positives
# - Medium (0.5-0.6): Balanced - RECOMMENDED
# - Higher (0.7-0.9): Only very confident detections
DEFAULT_CONFIDENCE = 0.5

# ==================================================
# RECOMMENDATIONS BY USE CASE
# ==================================================

# FOR REAL-TIME WEBCAM (Smooth performance):
# PROCESS_EVERY_N_FRAMES = 2
# SCALE_FACTOR = 0.75
# JPEG_QUALITY = 70

# FOR HIGH-QUALITY VIDEO ANALYSIS (Accurate but slower):
# PROCESS_EVERY_N_FRAMES = 1
# SCALE_FACTOR = 1.0
# JPEG_QUALITY = 85

# FOR MAXIMUM SPEED (Fast but lower quality):
# PROCESS_EVERY_N_FRAMES = 3
# SCALE_FACTOR = 0.5
# JPEG_QUALITY = 60
