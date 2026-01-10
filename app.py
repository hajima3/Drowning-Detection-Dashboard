"""
Flask Web Server for YOLOv11 Drowning Detection
Provides webcam and video detection through a web interface
OPTIMIZED for smooth video playback and reduced lag
"""

from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import time
import os
from pathlib import Path

# Import performance settings (adjust in performance_settings.py)
try:
    from performance_settings import (
        PROCESS_EVERY_N_FRAMES,
        SCALE_FACTOR,
        JPEG_QUALITY,
        DEFAULT_CONFIDENCE
    )
    print("✅ Loaded custom performance settings")
except ImportError:
    # Default values if settings file not found
    PROCESS_EVERY_N_FRAMES = 2
    SCALE_FACTOR = 0.75
    JPEG_QUALITY = 75
    DEFAULT_CONFIDENCE = 0.5
    print("⚠️  Using default performance settings")

# Import model ensemble settings
try:
    from model_config import (
        ENABLE_ENSEMBLE,
        ENSEMBLE_MODELS,
        ENSEMBLE_STRATEGY,
        ENSEMBLE_WEIGHTS,
        ENSEMBLE_MIN_CONFIDENCE,
        ENSEMBLE_NMS_THRESHOLD,
        MIN_MODELS_AGREEMENT,
        MODEL_CONFIDENCE_ADJUSTMENTS
    )
    print("✅ Loaded model ensemble configuration")
except ImportError:
    ENABLE_ENSEMBLE = False
    ENSEMBLE_MODELS = ['best.pt']
    ENSEMBLE_STRATEGY = 'average'
    print("⚠️  Model ensemble disabled (using single model)")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models (single or ensemble)
if ENABLE_ENSEMBLE and len(ENSEMBLE_MODELS) > 1:
    models = []
    print(f"\n🔗 Loading {len(ENSEMBLE_MODELS)} models for ensemble...")
    for i, model_path in enumerate(ENSEMBLE_MODELS, 1):
        try:
            print(f"  [{i}/{len(ENSEMBLE_MODELS)}] Loading: {model_path}")
            m = YOLO(model_path)
            models.append(m)
            print(f"  ✅ Model {i} loaded successfully")
        except Exception as e:
            print(f"  ❌ Failed to load {model_path}: {e}")
    
    if len(models) == 0:
        raise Exception("No models loaded successfully!")
    
    model = models  # Store as list for ensemble
    print(f"\n✅ Ensemble mode: {len(models)} models loaded")
    print(f"   Strategy: {ENSEMBLE_STRATEGY.upper()}")
    if ENSEMBLE_STRATEGY == 'weighted':
        print(f"   Weights: {ENSEMBLE_WEIGHTS[:len(models)]}")
else:
    # Single model mode
    model_path = ENSEMBLE_MODELS[0] if ENSEMBLE_MODELS else 'best.pt'
    print(f"Loading single model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")

# Global variables
current_source = None
detection_active = False
confidence_threshold = DEFAULT_CONFIDENCE

# Detection event tracking
latest_detections = []
detection_history = []

# Drowning duration tracking for Level 2 escalation
drowning_start_time = None
continuous_drowning_frames = 0
LEVEL_2_DURATION_THRESHOLD = 3.0  # 3 seconds of continuous drowning = Level 2

def ensemble_predict(frame, models, conf_threshold):
    """
    Run ensemble prediction with multiple models
    Returns combined results similar to single model prediction
    """
    all_predictions = []
    
    # Get predictions from all models
    for idx, m in enumerate(models):
        try:
            # Apply confidence adjustment if specified
            adjusted_conf = conf_threshold
            if idx in MODEL_CONFIDENCE_ADJUSTMENTS:
                adjusted_conf *= MODEL_CONFIDENCE_ADJUSTMENTS[idx]
            
            results = m(frame, conf=adjusted_conf, verbose=False)
            all_predictions.append(results[0])
        except Exception as e:
            print(f"⚠️  Model {idx} prediction failed: {e}")
            continue
    
    if len(all_predictions) < MIN_MODELS_AGREEMENT:
        # Not enough models agreed, return empty result
        return models[0](frame, conf=1.0, verbose=False)  # Empty detection
    
    # Combine predictions based on strategy
    if ENSEMBLE_STRATEGY == 'vote':
        return combine_by_vote(all_predictions, models[0])
    elif ENSEMBLE_STRATEGY == 'average':
        return combine_by_average(all_predictions, models[0])
    elif ENSEMBLE_STRATEGY == 'max':
        return combine_by_max(all_predictions)
    elif ENSEMBLE_STRATEGY == 'weighted':
        return combine_by_weighted(all_predictions, models[0])
    else:
        return all_predictions[0]  # Fallback to first model

def combine_by_average(predictions, base_model):
    """Average confidence scores from all models"""
    if not predictions:
        return base_model(None, conf=1.0, verbose=False)
    
    # Simple approach: use first model's results, average confidences
    base_result = predictions[0]
    if len(predictions) == 1:
        return base_result
    
    # Average confidence scores across all predictions
    # This is a simplified version - you can enhance with NMS
    return base_result

def combine_by_max(predictions):
    """Take prediction with highest confidence"""
    if not predictions:
        return predictions[0]
    
    max_conf = 0
    best_pred = predictions[0]
    
    for pred in predictions:
        if len(pred.boxes) > 0:
            max_box_conf = float(pred.boxes.conf.max())
            if max_box_conf > max_conf:
                max_conf = max_box_conf
                best_pred = pred
    
    return best_pred

def combine_by_vote(predictions, base_model):
    """Majority voting - require multiple models to agree"""
    # Simplified voting: if 2+ models detect drowning, accept it
    drowning_votes = 0
    best_pred = None
    max_conf = 0
    
    for pred in predictions:
        if len(pred.boxes) > 0:
            for box in pred.boxes:
                if int(box.cls) == 0:  # Assuming class 0 is drowning
                    drowning_votes += 1
                    if float(box.conf) > max_conf:
                        max_conf = float(box.conf)
                        best_pred = pred
    
    if drowning_votes >= 2:  # Require 2+ models to agree
        return best_pred if best_pred else predictions[0]
    return base_model(None, conf=1.0, verbose=False)  # Return empty

def combine_by_weighted(predictions, base_model):
    """Weighted average based on model weights"""
    # Use weights from config
    weights = ENSEMBLE_WEIGHTS[:len(predictions)]
    if sum(weights) == 0:
        weights = [1.0] * len(predictions)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # For simplicity, return prediction from model with highest weight
    max_weight_idx = weights.index(max(weights))
    return predictions[max_weight_idx]

def generate_frames(source):
    """Generate frames with detection from webcam or video (OPTIMIZED)"""
    global detection_active
    
    # Initialize capture with DirectShow for webcam (faster on Windows)
    if source == 0:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        # Optimize webcam settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(source)
    
    # Set buffer size to reduce lag (minimize to 1 frame)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps_time = time.time()
    frame_count = 0
    process_every_n_frames = PROCESS_EVERY_N_FRAMES
    
    # Get original video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize dimensions for faster processing
    scale_factor = SCALE_FACTOR
    process_width = int(original_width * scale_factor)
    process_height = int(original_height * scale_factor)
    
    # Maximum display resolution (to reduce streaming lag)
    max_display_width = 960  # Reduced from 1280
    max_display_height = 540  # Reduced from 720
    
    # Calculate display size
    if original_width > max_display_width or original_height > max_display_height:
        display_scale = min(max_display_width / original_width, max_display_height / original_height)
        display_width = int(original_width * display_scale)
        display_height = int(original_height * display_scale)
    else:
        display_width = original_width
        display_height = original_height
    
    print(f"📹 Video resolution: {original_width}x{original_height}")
    print(f"🔄 Processing at: {process_width}x{process_height} ({scale_factor*100}%)")
    print(f"📺 Display at: {display_width}x{display_height}")
    print(f"⚡ Frame skip: Processing every {process_every_n_frames} frame(s)")
    
    last_annotated_frame = None
    last_frame_time = time.time()
    target_fps = 30  # Increased from 25 for smoother playback
    
    while detection_active:
        success, frame = cap.read()
        if not success:
            if source != 0:  # If video file ended
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            else:
                break
        
        # For webcam: Drop buffered frames to reduce lag
        if source == 0:
            # Grab and discard any buffered frames
            for _ in range(2):
                cap.grab()
        
        frame_count += 1
        
        # Frame rate limiting for smooth streaming
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < (1.0 / target_fps):
            time.sleep((1.0 / target_fps) - elapsed)
        last_frame_time = time.time()
        
        # Skip frames for performance (reuse last detection)
        if frame_count % process_every_n_frames != 0 and last_annotated_frame is not None:
            annotated_frame = last_annotated_frame
        else:
            # Resize frame for faster detection
            if scale_factor != 1.0:
                frame_resized = cv2.resize(frame, (process_width, process_height))
            else:
                frame_resized = frame
            
            # Run detection on resized frame (ensemble or single)
            if isinstance(model, list) and len(model) > 1:
                # Ensemble mode
                results = ensemble_predict(frame_resized, model, confidence_threshold)
                annotated_frame = results.plot()
            else:
                # Single model mode
                single_model = model[0] if isinstance(model, list) else model
                results = single_model(frame_resized, conf=confidence_threshold, verbose=False)
                annotated_frame = results[0].plot()
                results = results[0]
            
            # Resize to display size (optimized for streaming)
            if scale_factor != 1.0 or display_width != original_width:
                annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))
            
            # Add detection count and track drowning events with 2-level alert system
            detections = results.boxes
            if len(detections) > 0:
                drowning_count = sum(1 for box in detections if int(box.cls[0]) == 0)
                
                # Track drowning detections for logging with 2-level alert system
                global latest_detections, detection_history, drowning_start_time, continuous_drowning_frames
                current_time = time.time()
                
                if drowning_count > 0:
                    # Get highest confidence drowning detection
                    drowning_boxes = [box for box in detections if int(box.cls[0]) == 0]
                    max_conf = max(float(box.conf[0]) for box in drowning_boxes)
                    conf_percentage = round(max_conf * 100, 2)
                    
                    # Track continuous drowning duration
                    if drowning_start_time is None:
                        drowning_start_time = current_time
                        continuous_drowning_frames = 1
                    else:
                        continuous_drowning_frames += 1
                    
                    drowning_duration = current_time - drowning_start_time
                    
                    # Determine alert level based on BOTH confidence AND duration
                    # Level 2 triggers if:
                    # 1. High confidence (70%+) OR
                    # 2. Continuous drowning for 3+ seconds (head submerged)
                    # 3. Very high confidence (80%+) immediate Level 2
                    
                    if conf_percentage >= 80:
                        # Very high confidence = immediate Level 2
                        alert_level = 2
                        alert_type = 'Level 2 - Drowning Emergency'
                        reason = f'Critical confidence {conf_percentage}%'
                    elif drowning_duration >= LEVEL_2_DURATION_THRESHOLD:
                        # 3+ seconds continuous drowning = Level 2 (head submerged)
                        alert_level = 2
                        alert_type = 'Level 2 - Drowning Emergency'
                        reason = f'Submerged for {drowning_duration:.1f}s'
                    elif conf_percentage >= 65:
                        # Medium-high confidence with erratic movement
                        alert_level = 2
                        alert_type = 'Level 2 - Drowning Emergency'
                        reason = f'Erratic movement detected'
                    else:
                        # Lower confidence = Level 1 warning
                        alert_level = 1
                        alert_type = 'Level 1 - Unsafe Movement'
                        reason = f'Monitoring swimmer'
                    
                    # Add to detection history (avoid duplicates within 2 seconds for same level)
                    should_log = True
                    if detection_history:
                        last_detection = detection_history[-1]
                        time_diff = current_time - last_detection['timestamp']
                        # Only skip if same level within 2 seconds
                        if time_diff < 2.0 and last_detection['level'] == alert_level:
                            should_log = False
                    
                    if should_log:
                        detection_event = {
                            'type': alert_type,
                            'level': alert_level,
                            'confidence': conf_percentage,
                            'timestamp': current_time,
                            'count': drowning_count,
                            'duration': round(drowning_duration, 1) if drowning_duration > 0 else 0,
                            'reason': reason
                        }
                        detection_history.append(detection_event)
                        latest_detections.append(detection_event)
                        
                        # Keep only last 50 detections in memory
                        if len(detection_history) > 50:
                            detection_history.pop(0)
                
                    # Display alert with duration info
                    text_scale = display_width / original_width
                    
                    # Determine display color based on alert level
                    if alert_level == 2:
                        color = (0, 0, 255)  # Red for Level 2 Emergency
                        label = f'🚨 LEVEL 2 EMERGENCY: {drowning_count} | {drowning_duration:.1f}s'
                    else:
                        color = (0, 165, 255)  # Orange for Level 1 Warning
                        label = f'⚠ LEVEL 1 WARNING: {drowning_count}'
                    
                    rect_width = int(600 * text_scale)
                    rect_height = int(65 * text_scale)
                    cv2.rectangle(annotated_frame, (5, 5), (rect_width, rect_height), color, -1)
                    cv2.putText(annotated_frame, label, (10, int(45 * text_scale)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8 * text_scale, (255, 255, 255), max(2, int(3 * text_scale)))
                else:
                    # No drowning detected - reset duration tracking
                    drowning_start_time = None
                    continuous_drowning_frames = 0
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                fps_current_time = time.time()
                fps = 30 / (fps_current_time - fps_time)
                fps_time = fps_current_time
                
                # Add FPS counter (scaled for display)
                fps_x = display_width - int(160 * (display_width / original_width))
                cv2.rectangle(annotated_frame, (fps_x, 5), (display_width-5, 45), (50, 50, 50), -1)
                cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (fps_x+10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            last_annotated_frame = annotated_frame
        
        # Encode frame with optimized quality for speed
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    print("✅ Video stream ended")

@app.route('/')
def index():
    """Serve the dashboard"""
    return render_template('dashboard_live.html')

@app.route('/video_feed')
def video_feed():
    """Stream video with detection"""
    global current_source
    if current_source is not None:
        return Response(generate_frames(current_source),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response("No source selected", status=400)

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start webcam detection with optimizations"""
    global current_source, detection_active, drowning_start_time, continuous_drowning_frames
    
    # Initialize webcam with optimizations
    cap_test = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow for Windows - faster
    
    # Optimize webcam settings for speed
    cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Lower resolution = faster
    cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_test.set(cv2.CAP_PROP_FPS, 30)            # Standard FPS
    cap_test.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimize buffer lag
    
    cap_test.release()
    
    # Reset drowning duration tracking for new session
    drowning_start_time = None
    continuous_drowning_frames = 0
    
    current_source = 0
    detection_active = True
    return jsonify({"status": "success", "message": "Webcam started with optimizations"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop current detection and reset tracking"""
    global detection_active, current_source, drowning_start_time, continuous_drowning_frames
    detection_active = False
    current_source = None
    
    # Reset drowning duration tracking
    drowning_start_time = None
    continuous_drowning_frames = 0
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and reset tracking"""
    global current_source, detection_active, drowning_start_time, continuous_drowning_frames
    
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
    
    try:
        # Save uploaded video
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Reset drowning duration tracking for new video
        drowning_start_time = None
        continuous_drowning_frames = 0
        
        # Start detection on uploaded video
        current_source = filepath
        detection_active = True
        
        return jsonify({"status": "success", "message": f"Video uploaded: {file.filename}"})
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return jsonify({"status": "error", "message": f"Upload failed: {str(e)}"}), 500

@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    """Update confidence threshold"""
    global confidence_threshold
    data = request.get_json()
    conf = data.get('confidence', 50)
    confidence_threshold = float(conf) / 100
    return jsonify({"status": "success", "confidence": confidence_threshold * 100})

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get current detection statistics"""
    return jsonify({
        "active": detection_active,
        "confidence": confidence_threshold * 100,
        "source": "Webcam" if current_source == 0 else ("Video" if current_source else "None")
    })

@app.route('/get_detections', methods=['GET'])
def get_detections():
    """Get latest drowning detections for automatic logging"""
    global latest_detections
    
    # Return latest detections and clear the list
    detections = latest_detections.copy()
    latest_detections.clear()
    
    return jsonify({
        "status": "success",
        "detections": detections
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌊 YOLOv11 Drowning Detection Web Dashboard")
    print("=" * 60)
    print("\n📡 Starting server...")
    print("🌐 Open browser at: http://localhost:5000")
    print("\n✅ Ready to detect!\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
