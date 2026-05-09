# imports
import os
import sys
import mlflow
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- 1) MLflow ----------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.start_run()
print("MLflow tracking to local './mlruns' directory.")

# ---------------- 2) Load data ----------------
DATA_PATH = "data/reviews.csv"

assert os.path.exists(DATA_PATH), "Missing data/reviews.csv. Please add the CSV."

df = pd.read_csv(DATA_PATH)
assert {"text","sentiment"} <= set(df.columns), "reviews.csv must have text,sentiment columns"
print(f"Dataset size: {len(df)} (pos={df.sentiment.sum()}, neg={len(df)-df.sentiment.sum()})")

# ---------------- 3) Strict no-leak split ----------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.25, random_state=42, stratify=df["sentiment"]
)
print(f"Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

# ---------------- 4) Validation lever: "C" controls model flexibility ----------------
C = 1.0  # default is 1.0
if len(sys.argv) > 1:
    try:
        C = float(sys.argv[1])  # e.g., python train.py 1.0  (likely PASS)
    except ValueError:
        print(f"Warning: invalid C '{sys.argv[1]}', using default {C}")
mlflow.log_param("C", C)  # keep runs auditable

# ---------------- 5) Train & Log (fit vectorizer ONLY on train) ----------------
pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), lowercase=True, min_df=1),
    LinearSVC(C=C, random_state=42)
)
pipe.fit(X_train_text, y_train)
preds = pipe.predict(X_test_text)

acc = accuracy_score(y_test, preds)
mlflow.log_metric("accuracy", acc)
print(f"Model Accuracy: {acc:.3f}")

# ---------------- 6) Gate ----------------
THRESH = 
if acc < THRESH:
    print(f"Validation Failed: Accuracy is below the {THRESH} threshold.")
    mlflow.end_run(status="FAILED")
    sys.exit(1)
else:
    print("Validation Passed: Accuracy is sufficient.")
    import mlflow.sklearn
    mlflow.sklearn.log_model(pipe, "sentiment-model")
    mlflow.end_run()
    sys.exit(0)
