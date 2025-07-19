import os
import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

import typer
import numpy as np
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

from settings import SAMPLE_LABELS, MAIN_OUTPUT_DIR
from utils import compute_metrics, print_details

OUTPUT_DIR = MAIN_OUTPUT_DIR + "tfidf"

app = typer.Typer()


@app.command()
def main(
    prefix: Optional[str] = typer.Option(
        None,
        "--prefix",
        "-p",
        help="If set, prepend this string to every example's text before training.",
    ),
):
    # Load datasets
    train_ds = Dataset.from_parquet("data/train.parquet")
    eval_ds = Dataset.from_parquet("data/eval.parquet")
    test_ds = Dataset.from_parquet("data/test.parquet")

    # Optionally add prefix
    if prefix is not None:

        def add_prefix(example):
            return {"text": f"{prefix}{example['text']}"}

        train_ds = train_ds.map(add_prefix)
        eval_ds = eval_ds.map(add_prefix)
        test_ds = test_ds.map(add_prefix)

    typer.echo("Datasets loaded")

    # Build a pipeline with TF-IDF vectorizer and logistic regression
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",  # default is without it, but this decreases the dictionary size significantly
                    min_df=0.02,  # Ignore terms that have a document frequency strictly lower than the given threshold. When float, proportion of docs.
                    max_df=0.95,  # ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
                    ngram_range=(1, 2),  # uni and bi-grams
                    max_features=50_000,
                    dtype=np.float32,  # Reduces the size of the resulting array without much quality sacrifice, default is float64
                ),
            ),
            ("clf", LogisticRegression()),
        ]
    )

    X_train_texts = train_ds["text"]
    y_train = train_ds["label"]
    X_test_texts = test_ds["text"]
    y_test = test_ds["label"]

    start_train = perf_counter()
    pipeline.fit(X_train_texts, y_train)
    end_train = perf_counter()
    train_time = end_train - start_train
    typer.echo(f"Training completed in {train_time:.3f} seconds")

    start_test = perf_counter()
    y_pred = pipeline.predict(X_test_texts)
    end_test = perf_counter()
    test_time = end_test - start_test
    typer.echo(f"Test run completed in {test_time:.3f} seconds")

    metrics = compute_metrics(y_pred, y_test)
    metrics["train_time_seconds"] = train_time
    metrics["test_time_seconds"] = test_time
    typer.echo(metrics)
    print_details(y_pred, y_test, SAMPLE_LABELS)

    typer.echo("Saving the model")
    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"tfidf_lr_{datetime_str}")
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(pipeline, os.path.join(save_path, "pipeline.joblib"))
    with open(os.path.join(save_path, f"metrics_{datetime_str}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    typer.echo("Model saved")

    # Example prediction
    test_example = test_ds[0]["text"]
    typer.echo(f"Example text: {test_example}")
    pred = pipeline.predict([test_example])[0]
    proba = pipeline.predict_proba([test_example])[0]
    typer.echo(f"Predicted class: {pred}")
    typer.echo(f"Predicted probabilities: {proba}")


if __name__ == "__main__":
    app()
