#!/usr/bin/env python3
"""Log pytest results and optional demo artifacts to MLflow."""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mlflow


def write_github_output(key: str, value: str) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output_file:
            output_file.write(f"{key}={value}\n")


def parse_pytest_xml(xml_path: Path) -> dict[str, int]:
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing pytest XML report: {xml_path}")

    root = ET.parse(xml_path).getroot()
    test_suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in test_suites:
        for key in totals:
            totals[key] += int(suite.attrib.get(key, 0))
    return totals


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}

    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def log_to_mlflow(args: argparse.Namespace) -> int:
    pytest_xml_path = Path(args.pytest_xml)
    metrics_path = Path(args.metrics_json)
    model_path = Path(args.model_dir)

    pytest_totals = parse_pytest_xml(pytest_xml_path)
    metrics = load_metrics(metrics_path)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("test_command", args.test_command)
        mlflow.log_param("pytest_xml", str(pytest_xml_path))

        for metric_name, metric_value in pytest_totals.items():
            mlflow.log_metric(f"pytest_{metric_name}", metric_value)
        mlflow.log_metric("pytest_exit_code", args.pytest_exit_code)

        if "dataset" in metrics:
            mlflow.log_param("dataset", metrics["dataset"])
        if "accuracy" in metrics:
            mlflow.log_metric("accuracy", float(metrics["accuracy"]))
            write_github_output("accuracy", str(metrics["accuracy"]))

        mlflow.log_artifact(str(pytest_xml_path), artifact_path="test-results")
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path), artifact_path="run-metadata")
        if model_path.exists():
            mlflow.log_artifacts(str(model_path), artifact_path="model")

    print(
        "Logged pytest results to MLflow: "
        f"tests={pytest_totals['tests']}, "
        f"failures={pytest_totals['failures']}, "
        f"errors={pytest_totals['errors']}, "
        f"skipped={pytest_totals['skipped']}, "
        f"exit_code={args.pytest_exit_code}"
    )
    return args.pytest_exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log pytest XML and optional artifacts to MLflow.")
    parser.add_argument("--pytest-xml", default="test-results/pytest.xml")
    parser.add_argument("--pytest-exit-code", type=int, default=0)
    parser.add_argument("--test-command", default="python -m pytest -s tests --junitxml=test-results/pytest.xml")
    parser.add_argument("--metrics-json", default="metrics.json")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    parser.add_argument("--experiment-name", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "ci-cd-demo"))
    parser.add_argument("--run-name", default="pytest-results")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(log_to_mlflow(parse_args()))
