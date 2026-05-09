#!/usr/bin/env python3
"""Fake training script for a CI/CD validation demo."""
import argparse
import json
from pathlib import Path


ACCURACY_BY_DATASET = {
    "big": 0.95,
    "small": 0.75,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create demo model metrics.")
    parser.add_argument("--dataset", choices=sorted(ACCURACY_BY_DATASET), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accuracy = ACCURACY_BY_DATASET[args.dataset]

    metrics = {
        "dataset": args.dataset,
        "accuracy": accuracy,
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.txt").write_text(
        f"Demo model artifact for dataset={args.dataset}, accuracy={accuracy}\n",
        encoding="utf-8",
    )

    print(f"Trained demo model for dataset={args.dataset} with accuracy={accuracy}")


if __name__ == "__main__":
    main()
