"""
Model Manager -- update deployed model info in model_metrics.json

Usage:
  python model_manager.py                        # Show current model info
  python model_manager.py deploy best.pt "note"  # Mark a model as deployed
"""

import json
from datetime import datetime
from pathlib import Path

METRICS_FILE = Path(__file__).parent / "model_metrics.json"


def load_metrics():
    with open(METRICS_FILE, 'r') as f:
        return json.load(f)


def save_metrics(data):
    with open(METRICS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def show_info():
    data = load_metrics()
    print("\n=== Deployed Model ===")
    print(f"  File:    {data.get('current_model', 'unknown')}")
    print(f"  Classes: {', '.join(data.get('classes', []))}")
    print(f"  Date:    {data.get('deployed_date', 'unknown')}")
    print(f"  Notes:   {data.get('notes', '')}")
    print()


def deploy(model_name, notes=""):
    data = load_metrics()
    data["current_model"] = model_name
    data["deployed_date"] = datetime.now().strftime("%Y-%m-%d")
    data["notes"] = notes
    save_metrics(data)
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
