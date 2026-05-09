#!/usr/bin/env python3
"""
Logs metrics.json + model/ to MLflow and exposes "accuracy" to GitHub Actions.
Uses local file store (./mlruns) by default.
"""
import json
import os
import sys
from pathlib import Path

import mlflow


def write_github_output(key: str, value: str) -> None:
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as output_file:
            output_file.write(f"{key}={value}\n")


def main() -> None:
    metrics_path = Path("metrics.json")
    if not os.path.exists(metrics_path):
        print(f"[log_to_mlflow.py] ERROR: {metrics_path} not found. Run train.py first.", file=sys.stderr)
        sys.exit(1)

    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    dataset = metrics.get("dataset", "unknown")
    accuracy = float(metrics.get("accuracy", 0.0))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("ci-cd-demo")

    with mlflow.start_run(run_name=f"demo-{dataset}"):
        mlflow.log_param("dataset", dataset)
        mlflow.log_metric("accuracy", accuracy)
        if Path("model").exists():
            mlflow.log_artifacts("model", artifact_path="model")

    write_github_output("accuracy", str(accuracy))
    print(f"Logged dataset={dataset}, accuracy={accuracy}")


if __name__ == "__main__":
    main()
