"""
Utility Functions
=================
Helper functions used across the ChurnGuard AI pipeline.
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime


def load_config(config_path="config/config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(results, filepath):
    """Save results dictionary to JSON file."""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Results saved to {filepath}")


def print_section_header(title, width=50):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def calculate_business_metrics(y_true, y_pred, avg_customer_value=500):
    """Calculate business-oriented churn metrics."""
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "total_customers": len(y_true),
        "actual_churners": int(y_true.sum()),
        "predicted_churners": int(y_pred.sum()),
        "correctly_identified": int(tp),
        "missed_churners": int(fn),
        "false_alarms": int(fp),
        "potential_revenue_saved": int(tp) * avg_customer_value,
        "revenue_at_risk": int(fn) * avg_customer_value,
        "retention_opportunity_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
    }
    return metrics


def format_percentage(value):
    """Format a float as a percentage string."""
    return f"{value:.1%}"


def log_experiment(model_name, metrics, filepath="logs/experiments.json"):
    """Log experiment results with timestamp."""
    ensure_directory(os.path.dirname(filepath))

    entry = {
        "timestamp": get_timestamp(),
        "model": model_name,
        "metrics": metrics,
    }

    # Load existing experiments or create new list
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            experiments = json.load(f)
    else:
        experiments = []

    experiments.append(entry)

    with open(filepath, "w") as f:
        json.dump(experiments, f, indent=2, default=str)

    print(f"📝 Experiment logged: {model_name} @ {entry['timestamp']}")
