"""
YOLOv11 Model Inference Module
Handles model loading, prediction, and ensemble operations
"""
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Union
import numpy as np


class ModelInference:
    """Handles YOLOv11 model inference and ensemble predictions"""
    
    def __init__(self, model_paths: Union[str, List[str]], 
                 enable_ensemble: bool = False,
                 ensemble_strategy: str = 'average',
                 ensemble_weights: Optional[List[float]] = None):
        """
        Initialize model inference
        
        Args:
            model_paths: Single path or list of paths to model files
            enable_ensemble: Whether to enable ensemble mode
            ensemble_strategy: Strategy for combining predictions (average, max, vote, weighted)
            ensemble_weights: Weights for weighted ensemble strategy
        """
        self.enable_ensemble = enable_ensemble and isinstance(model_paths, list) and len(model_paths) > 1
        self.ensemble_strategy = ensemble_strategy
        self.ensemble_weights = ensemble_weights or []
        
        # Load models
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        
        self.models = []
        self._load_models(model_paths)
    
    def _load_models(self, model_paths: List[str]):
        """Load YOLOv11 models from paths"""
        print(f"\n🔧 Loading {len(model_paths)} model(s)...")
        
        for i, path in enumerate(model_paths, 1):
            try:
                print(f"  [{i}/{len(model_paths)}] Loading: {path}")
                model = YOLO(path)
                self.models.append(model)
                print(f"  ✅ Model {i} loaded successfully")
            except Exception as e:
                print(f"  ❌ Failed to load {path}: {e}")
        
        if len(self.models) == 0:
            raise Exception("No models loaded successfully!")
        
        if self.enable_ensemble:
            print(f"\n✅ Ensemble mode: {len(self.models)} models")
            print(f"   Strategy: {self.ensemble_strategy.upper()}")
        else:
            print(f"\n✅ Single model mode")
    
    def predict(self, frame, conf_threshold: float = 0.5, verbose: bool = False):
        """
        Run inference on frame
        
        Args:
            frame: Input frame (numpy array)
            conf_threshold: Confidence threshold for detections
            verbose: Whether to print verbose output
        
        Returns:
            Detection results (ultralytics Results object)
        """
        if self.enable_ensemble and len(self.models) > 1:
            return self._ensemble_predict(frame, conf_threshold, verbose)
        else:
            # Single model prediction
            model = self.models[0]
            results = model(frame, conf=conf_threshold, verbose=verbose)
            return results[0]
    
    def _ensemble_predict(self, frame, conf_threshold: float, verbose: bool):
        """Run ensemble prediction with multiple models"""
        all_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            try:
                results = model(frame, conf=conf_threshold, verbose=verbose)
                all_predictions.append(results[0])
            except Exception as e:
                print(f"⚠️  Model prediction failed: {e}")
                continue
        
        if len(all_predictions) == 0:
            # Return empty result
            return self.models[0](frame, conf=1.0, verbose=False)[0]
        
        # Combine predictions based on strategy
        if self.ensemble_strategy == 'vote':
            return self._combine_by_vote(all_predictions)
        elif self.ensemble_strategy == 'average':
            return self._combine_by_average(all_predictions)
        elif self.ensemble_strategy == 'max':
            return self._combine_by_max(all_predictions)
        elif self.ensemble_strategy == 'weighted':
            return self._combine_by_weighted(all_predictions)
        else:
            return all_predictions[0]
    
    def _combine_by_average(self, predictions):
        """Average confidence scores from all models"""
        # Simplified: use first model's results
        # TODO: Implement proper confidence averaging with NMS
        return predictions[0]
    
    def _combine_by_max(self, predictions):
        """Take prediction with highest confidence"""
        max_conf = 0
        best_pred = predictions[0]
        
        for pred in predictions:
            if len(pred.boxes) > 0:
                max_box_conf = float(pred.boxes.conf.max())
                if max_box_conf > max_conf:
                    max_conf = max_box_conf
                    best_pred = pred
        
        return best_pred
    
    def _combine_by_vote(self, predictions):
        """Majority voting - require multiple models to agree"""
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
        
        # Require at least 2 models to agree
        if drowning_votes >= 2 and best_pred:
            return best_pred
        
        # Return empty result if not enough agreement
        return self.models[0](None, conf=1.0, verbose=False)[0]
    
    def _combine_by_weighted(self, predictions):
        """Weighted average based on model weights"""
        weights = self.ensemble_weights[:len(predictions)]
        if sum(weights) == 0:
            weights = [1.0] * len(predictions)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Return prediction from model with highest weight
        max_weight_idx = weights.index(max(weights))
        return predictions[max_weight_idx]
    
    def get_model_count(self) -> int:
        """Get number of loaded models"""
        return len(self.models)
    
    def is_ensemble(self) -> bool:
        """Check if ensemble mode is enabled"""
        return self.enable_ensemble
