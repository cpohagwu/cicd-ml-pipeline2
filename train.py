import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "sentiment-ci")

MIN_ACCURACY = 0.60
MAX_INFERENCE_COST = 0.003  # USD per 1,000 predictions

TEST_SIZE = 0.25
RANDOM_STATE = 42
NGRAM_RANGE = (1, 2)
LOWERCASE = True
MIN_DF = 1


def resolve_data_path() -> Path:
    """Prefer the course data layout, but support the current repo layout."""
    candidates = [
        Path(os.environ.get("DATA_PATH", "")) if os.environ.get("DATA_PATH") else None,
        Path("data/reviews.csv"),
        Path("reviews.csv"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError("Missing reviews data. Expected DATA_PATH, data/reviews.csv, or reviews.csv.")


def parse_c(default: float = 1.0) -> float:
    if len(sys.argv) <= 1:
        return default
    try:
        return float(sys.argv[1])
    except ValueError:
        print(f"Warning: invalid C '{sys.argv[1]}', using default {default}")
        return default


def parse_inference_cost(default: float = 0.001) -> float:
    raw_cost = os.environ.get("INFERENCE_COST_PER_1000", str(default))
    try:
        return float(raw_cost)
    except ValueError:
        print(f"Warning: invalid INFERENCE_COST_PER_1000 '{raw_cost}', using default {default}")
        return default


def main() -> int:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_path = resolve_data_path()
    c_value = parse_c()
    inference_cost = parse_inference_cost()

    with mlflow.start_run(run_name="sentiment-validation-gate") as run:
        print(f"MLflow tracking to local './mlruns' directory. run_id={run.info.run_id}")

        df = pd.read_csv(data_path)
        assert {"text", "sentiment"} <= set(df.columns), "reviews data must have text,sentiment columns"
        positive_count = int(df.sentiment.sum())
        negative_count = int(len(df) - positive_count)
        print(f"Dataset size: {len(df)} (pos={positive_count}, neg={negative_count})")

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            df["text"],
            df["sentiment"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df["sentiment"],
        )
        print(f"Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

        mlflow.log_params(
            {
                "C": c_value,
                "data_path": str(data_path),
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "vectorizer_ngram_range": str(NGRAM_RANGE),
                "vectorizer_lowercase": LOWERCASE,
                "vectorizer_min_df": MIN_DF,
                "min_accuracy": MIN_ACCURACY,
                "max_inference_cost": MAX_INFERENCE_COST,
                "inference_cost_units": "USD per 1,000 predictions",
            }
        )

        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=NGRAM_RANGE, lowercase=LOWERCASE, min_df=MIN_DF),
            LinearSVC(C=c_value, random_state=RANDOM_STATE),
        )
        pipe.fit(X_train_text, y_train)
        preds = pipe.predict(X_test_text)

        accuracy = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("inference_cost", inference_cost)
        print(f"Model Accuracy: {accuracy:.3f}")
        print(f"Inference Cost: ${inference_cost:.6f} per 1,000 predictions")

        run_metadata = {
            "run_id": run.info.run_id,
            "experiment_name": EXPERIMENT_NAME,
            "data_path": str(data_path),
            "dataset_rows": len(df),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "train_rows": len(X_train_text),
            "test_rows": len(X_test_text),
            "parameters": {
                "C": c_value,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "ngram_range": NGRAM_RANGE,
                "lowercase": LOWERCASE,
                "min_df": MIN_DF,
            },
            "metrics": {
                "accuracy": accuracy,
                "inference_cost": inference_cost,
            },
            "thresholds": {
                "minimum_accuracy": MIN_ACCURACY,
                "maximum_inference_cost": MAX_INFERENCE_COST,
            },
        }
        mlflow.log_dict(run_metadata, "run_metadata.json")
        mlflow.sklearn.log_model(pipe, "sentiment-model")

        failed_checks = []
        if accuracy < MIN_ACCURACY:
            failed_checks.append(f"accuracy {accuracy:.3f} < {MIN_ACCURACY:.3f}")
        if inference_cost > MAX_INFERENCE_COST:
            failed_checks.append(f"inference_cost {inference_cost:.6f} > {MAX_INFERENCE_COST:.6f}")

        if failed_checks:
            print("Validation Failed: " + "; ".join(failed_checks))
            mlflow.set_tag("validation_status", "failed")
            raise SystemExit(1)

        print("Validation Passed: accuracy and inference cost meet thresholds.")
        mlflow.set_tag("validation_status", "passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
