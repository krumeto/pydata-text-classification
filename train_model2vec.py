import os
import json
from datetime import datetime, timezone
import typer
from time import perf_counter
from datasets import Dataset
from model2vec.train import StaticModelForClassification
from typing import Optional
from huggingface_hub import login
from dotenv import load_dotenv

from settings import SAMPLE_LABELS, MAIN_OUTPUT_DIR
from utils import compute_metrics, print_details

OUTPUT_DIR = MAIN_OUTPUT_DIR + "model2vec"

app = typer.Typer()


@app.command()
def main(
    model_name: str = typer.Argument(
        ...,
        help="Pre-trained model2vec checkpoint name (e.g., minishlab/potion-base-32m)",
    ),
    prefix: Optional[str] = typer.Option(
        None,
        "--prefix",
        "-p",
        help="Optionally prepend this string to every example's text before training.",
    ),
):
    """
    Train a model2vec classifier and compute metrics aligned with the SetFit baseline.
    """
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

    if prefix:

        def add_prefix(ex):
            ex["text"] = prefix + ex["text"]
            return ex

        train_ds = train_ds.map(add_prefix)
        eval_ds = eval_ds.map(add_prefix)
        test_ds = test_ds.map(add_prefix)

    typer.echo("Datasets loaded")

    typer.echo(f"Loading model2vec classifier {model_name}")
    classifier = StaticModelForClassification.from_pretrained(model_name=model_name)
    typer.echo("Classifier loaded")

    texts = train_ds["text"]
    labels = train_ds["label"]
    typer.echo("Starting training...")
    start_train = perf_counter()
    classifier = classifier.fit(texts, labels)
    end_train = perf_counter()
    train_time = end_train - start_train
    typer.echo(f"Training completed in {train_time:.3f} seconds")

    typer.echo("Evaluating on test set...")
    start_test = perf_counter()
    y_pred_test = classifier.predict(test_ds["text"])
    end_test = perf_counter()
    test_time = end_test - start_test
    typer.echo(f"Test run completed in {test_time:.3f} seconds")

    y_true_test = test_ds["label"]
    test_metrics = compute_metrics(y_pred_test, y_true_test)
    test_metrics["train_time_seconds"] = train_time
    test_metrics["test_time_seconds"] = test_time
    typer.echo("Test metrics:")
    typer.echo(test_metrics)

    print_details(y_pred_test, y_true_test, SAMPLE_LABELS)

    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    project_name = os.getenv("PROJECT_NAME", "text-class-tutorial")
    hf_profile = os.getenv("HF_PROFILE_NAME", "user")
    repo_name = f"{hf_profile}/{project_name}-model2vec"
    typer.echo(f"Pushing model to Hugging Face Hub as: {repo_name}")

    pipeline = classifier.to_pipeline()
    pipeline.push_to_hub(repo_name)
    typer.echo(f"Model successfully pushed to hub: {repo_name}")

    # Save metrics locally only
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{datetime_str}.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    typer.echo(f"Metrics saved locally to: {metrics_path}")

    # Example prediction
    test_example = test_ds[0]["text"]
    typer.echo(f"Example text: {test_example}")
    pred = pipeline.predict([test_example])[0]
    proba = pipeline.predict_proba([test_example])[0]
    typer.echo(f"Predicted class: {pred}")
    typer.echo(f"Predicted probabilities: {proba}")


if __name__ == "__main__":
    app()
