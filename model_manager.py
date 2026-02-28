"""
Model Manager — view or update deployed model info in model_metrics.json

Usage:
  python model_manager.py                        # Show current model info
  python model_manager.py deploy best.pt "note"  # Mark a model as deployed
"""

import json
from datetime import datetime
from pathlib import Path

METRICS_FILE = Path(__file__).parent / "model_metrics.json"


def load_metrics():
    with open(METRICS_FILE) as f:
        return json.load(f)


def save_metrics(data):
    with open(METRICS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def show_info():
    d = load_metrics()
    print()
    print("=== Deployed Model ===")
    print(f"  File    : {d.get('current_model', 'unknown')}")
    print(f"  Classes : {', '.join(d.get('classes', []))}")
    print(f"  Date    : {d.get('deployed_date', 'unknown')}")
    training = d.get('training', {})
    if training:
        print(f"  mAP50   : {training.get('mAP50', 'N/A')}")
        print(f"  Recall  : {training.get('recall', 'N/A')}")
    print()


def deploy(model_name, notes=""):
    d = load_metrics()
    d["current_model"] = model_name
    d["deployed_date"] = datetime.now().strftime("%Y-%m-%d")
    if notes:
        d["notes"] = notes
    save_metrics(d)
    print(f"Deployed: {model_name}")
    show_info()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        show_info()
    elif sys.argv[1] == "deploy" and len(sys.argv) >= 3:
        deploy(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        print(__doc__)

