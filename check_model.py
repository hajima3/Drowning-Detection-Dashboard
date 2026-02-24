"""
Check best.pt model file for classes and accuracy metrics
"""
from ultralytics import YOLO
import json

def check_model(model_path='best.pt'):
    print("\n" + "="*60)
    print("YOLO MODEL INSPECTION - best.pt")
    print("="*60)
    
    # Load the model
    model = YOLO(model_path)
    
    # 1. Classes
    print("\n📋 MODEL CLASSES:")
    print("-" * 40)
    names = model.names
    print(f"Total Classes: {len(names)}")
    for idx, name in names.items():
        print(f"  Class {idx}: {name}")
    
    # 2. Check class mapping consistency
    print("\n🔍 CLASS MAPPING CHECK:")
    print("-" * 40)
    expected_classes = ["Level 0", "Level 1"]
    actual_classes = list(names.values())
    
    if actual_classes == expected_classes:
        print("✅ Classes match expected mapping in model_metrics.json")
    else:
        print("⚠️  Classes DO NOT match!")
        print(f"   Expected: {expected_classes}")
        print(f"   Actual:   {actual_classes}")
    
    # 3. Training Results
    print("\n📊 TRAINING METRICS:")
    print("-" * 40)
    
    try:
        # Try to get training results from checkpoint
        if hasattr(model, 'ckpt'):
            ckpt = model.ckpt
            
            # Look for key metrics in different possible locations
            if 'best_fitness' in ckpt:
                print(f"Best Fitness: {ckpt['best_fitness']}")
            
            if 'epoch' in ckpt:
                print(f"Total Epochs: {ckpt['epoch']}")
            
            # Extract final metrics
            results_found = False
            
            # Check for results in checkpoint
            for key in ['train_results', 'results', 'metrics']:
                if key in ckpt:
                    print(f"\nFound {key}:")
                    results_found = True
                    # Get last epoch results
                    data = ckpt[key]
                    if isinstance(data, dict):
                        for metric, values in data.items():
                            if isinstance(values, list) and len(values) > 0:
                                # Get last epoch value
                                last_val = values[-1]
                                print(f"  {metric}: {last_val}")
                    break
            
            if not results_found:
                print("⚠️  No detailed metrics found in checkpoint")
                print(f"Available checkpoint keys: {list(ckpt.keys())}")
                
                # Try to extract from results.csv if it exists
                print("\n💡 Attempting to read results from training...")
                
    except Exception as e:
        print(f"❌ Error reading training metrics: {e}")
    
    # 4. Model Info
    print("\n🔧 MODEL INFORMATION:")
    print("-" * 40)
    print(f"Model Type: {type(model.model).__name__}")
    print(f"Task: {model.task}")
    
    # Try to get model parameters
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"Total Parameters: {total_params:,}")
    except:
        pass
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_model()
