import os
import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional
from pathlib import Path
from tempfile import mkdtemp, mkstemp
import pickle

import typer
import numpy as np
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from skops import card, hub_utils
from huggingface_hub import login, create_repo
from dotenv import load_dotenv

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
    load_dotenv()

    train_ds = Dataset.from_parquet("data/train.parquet")
    eval_ds = Dataset.from_parquet("data/eval.parquet")
    test_ds = Dataset.from_parquet("data/test.parquet")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, write_permission=True)
        typer.echo("Authenticated with Hugging Face Hub")
    else:
        typer.echo(
            "Warning: HF_TOKEN not found in environment, attempting manual login"
        )
        login()

    if prefix is not None:

        def add_prefix(example):
            return {"text": f"{prefix}{example['text']}"}

        train_ds = train_ds.map(add_prefix)
        eval_ds = eval_ds.map(add_prefix)
        test_ds = test_ds.map(add_prefix)

    typer.echo("Datasets loaded")

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

    # Save model to Hugging Face Hub using skops and metrics locally
    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    project_name = os.getenv("PROJECT_NAME", "text-class-tutorial")
    hf_profile = os.getenv("HF_PROFILE_NAME", "user")
    repo_name = f"{hf_profile}/{project_name}-tfidf"
    typer.echo(f"Pushing model to Hugging Face Hub as: {repo_name}")

    # Create temporary files for skops
    _, pkl_name = mkstemp(prefix="skops-", suffix=".pkl")
    with open(pkl_name, mode="bw") as f:
        pickle.dump(pipeline, file=f)

    local_repo = mkdtemp(prefix="skops-")

    # Initialize skops hub utils
    hub_utils.init(
        model=pkl_name,
        requirements=["scikit-learn", "numpy"],
        dst=local_repo,
        task="text-classification",
        data=X_test_texts[:100],  # Use first 100 examples for model card
    )

    # Create model card
    model_card = card.Card(
        pipeline, metadata=card.metadata_from_config(Path(local_repo))
    )
    model_card.metadata.license = "mit"

    # Add model details
    model_description = (
        "This is a TF-IDF + Logistic Regression model trained for text classification. "
        "It uses TF-IDF vectorization with uni and bi-grams, followed by logistic regression."
    )
    limitations = "This model is for demonstration purposes."
    get_started_code = (
        "import pickle\n"
        "with open('model.pkl', 'rb') as file:\n"
        "    pipeline = pickle.load(file)\n"
        "prediction = pipeline.predict(['your text here'])"
    )

    model_card.add(
        model_description=model_description,
        limitations=limitations,
        get_started_code=get_started_code,
    )

    model_card.add_metrics(**metrics)

    model_card.save(Path(local_repo) / "README.md")

    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=repo_name, token=hf_token, exist_ok=True)
        typer.echo(f"Repository {repo_name} created/verified")
    except Exception as e:
        typer.echo(f"Repository creation error (may already exist): {e}")

    hub_utils.push(
        repo_id=repo_name,
        source=local_repo,
        token=hf_token,
        commit_message="Upload TF-IDF model using skops",
    )
    typer.echo(f"Model successfully pushed to hub: {repo_name}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{datetime_str}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    typer.echo(f"Metrics saved locally to: {metrics_path}")

    # Clean up temporary files
    os.unlink(pkl_name)
    import shutil

    shutil.rmtree(local_repo)

    # Example prediction
    test_example = test_ds[0]["text"]
    typer.echo(f"Example text: {test_example}")
    pred = pipeline.predict([test_example])[0]
    proba = pipeline.predict_proba([test_example])[0]
    typer.echo(f"Predicted class: {pred}")
    typer.echo(f"Predicted probabilities: {proba}")


if __name__ == "__main__":
    app()
