"""
Model Iteration Manager
Track model performance across training iterations
"""

import json
from datetime import datetime
from pathlib import Path

METRICS_FILE = Path(__file__).parent / "model_metrics.json"

def load_metrics():
    """Load model metrics from JSON file"""
    with open(METRICS_FILE, 'r') as f:
        return json.load(f)

def save_metrics(data):
    """Save model metrics to JSON file"""
    with open(METRICS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def add_iteration(iteration_num, model_name, dataset_size, precision, recall, f1_score, map50, notes=""):
    """Add a new training iteration"""
    data = load_metrics()
    
    new_iteration = {
        "iteration": iteration_num,
        "model_name": model_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "dataset_size": dataset_size,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "map50": round(map50, 2),
        "status": "deployed" if f1_score >= data["target_f1"] else "needs_improvement",
        "notes": notes
    }
    
    # Check if iteration already exists, update it
    existing = next((i for i in data["iterations"] if i["iteration"] == iteration_num), None)
    if existing:
        idx = data["iterations"].index(existing)
        data["iterations"][idx] = new_iteration
    else:
        data["iterations"].append(new_iteration)
    
    # Update current model if this iteration meets target
    if f1_score >= data["target_f1"]:
        data["current_model"] = model_name
    
    save_metrics(data)
    print(f"✅ Iteration {iteration_num} saved: F1={f1_score}% | Status: {new_iteration['status']}")
    return new_iteration

def get_current_iteration():
    """Get the current iteration number"""
    data = load_metrics()
    return len(data["iterations"])

def get_best_model():
    """Get the iteration with best F1 score"""
    data = load_metrics()
    if not data["iterations"]:
        return None
    return max(data["iterations"], key=lambda x: x["f1_score"])

def display_summary():
    """Display summary of all iterations"""
    data = load_metrics()
    
    print("\n" + "="*70)
    print("📊 MODEL ITERATION SUMMARY")
    print("="*70)
    print(f"Target F1 Score: {data['target_f1']}%")
    print(f"Current Model: {data['current_model']}")
    print(f"Total Iterations: {len(data['iterations'])}\n")
    
    for it in data["iterations"]:
        status_emoji = "✅" if it["status"] == "deployed" else "🔄" if it["status"] == "needs_improvement" else "⏳"
        print(f"{status_emoji} Iteration {it['iteration']}: {it['model_name']}")
        print(f"   Date: {it['date']} | Dataset: {it['dataset_size']} samples")
        print(f"   Precision: {it['precision']}% | Recall: {it['recall']}% | F1: {it['f1_score']}% | mAP50: {it['map50']}%")
        if it['notes']:
            print(f"   Notes: {it['notes']}")
        print()
    
    best = get_best_model()
    if best:
        print(f"🏆 Best Model: Iteration {best['iteration']} (F1: {best['f1_score']}%)")
    print("="*70 + "\n")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("📋 Usage:")
        print("  python model_manager.py add <iteration> <precision> <recall> <f1> <map50> <dataset_size> [notes]")
        print("  python model_manager.py summary")
        print("\nExample:")
        print('  python model_manager.py add 1 96.5 97.2 96.8 96.9 500 "Initial model"')
        display_summary()
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "summary":
        display_summary()
    
    elif command == "add" and len(sys.argv) >= 7:
        iteration = int(sys.argv[2])
        precision = float(sys.argv[3])
        recall = float(sys.argv[4])
        f1 = float(sys.argv[5])
        map50 = float(sys.argv[6])
        dataset_size = int(sys.argv[7]) if len(sys.argv) > 7 else 0
        notes = sys.argv[8] if len(sys.argv) > 8 else ""
        
        model_name = f"best_v{iteration}.pt"
        add_iteration(iteration, model_name, dataset_size, precision, recall, f1, map50, notes)
        display_summary()
    
    else:
        print("❌ Invalid command. Use 'summary' or 'add' with required parameters.")
