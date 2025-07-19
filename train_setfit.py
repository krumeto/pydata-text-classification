import os
import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

import typer
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from huggingface_hub import login
from dotenv import load_dotenv

from settings import SAMPLE_LABELS, MAIN_OUTPUT_DIR
from utils import compute_metrics, print_details

OUTPUT_DIR = MAIN_OUTPUT_DIR + "setfit"

app = typer.Typer()


@app.command()
def main(
    model_name: str,
    prefix: Optional[str] = typer.Option(
        None,
        "--prefix",
        "-p",
        help="If set, prepend this string to every example's text before training.",
    ),
):
    load_dotenv()

    train_dataset = Dataset.from_parquet("data/train.parquet")
    eval_dataset = Dataset.from_parquet("data/eval.parquet")
    test_dataset = Dataset.from_parquet("data/test.parquet")

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

        train_dataset = train_dataset.map(add_prefix)
        eval_dataset = eval_dataset.map(add_prefix)
        test_dataset = test_dataset.map(add_prefix)

    print("Datasets loaded")

    model = SetFitModel.from_pretrained(
        f"{model_name}",
        labels=SAMPLE_LABELS,
    )
    typer.echo("Model loaded")
    model.model_body.max_seq_length = 512

    typer.echo(f"{model_name} successfully initialized!")
    args = TrainingArguments(
        batch_size=64,
        max_steps=90,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric=compute_metrics,
        column_mapping={
            "text": "text",
            "label": "label",
        }, 
    )

    # Train and evaluate
    start_train = perf_counter()
    trainer.train()
    end_train = perf_counter()
    train_time = end_train - start_train
    typer.echo(f"Training completed in {train_time:.3f} seconds")

    start_test = perf_counter()
    metrics = trainer.evaluate(test_dataset)
    end_test = perf_counter()
    test_time = end_test - start_test
    typer.echo(f"Test run completed in {test_time:.3f} seconds")

    metrics["train_time_seconds"] = train_time
    metrics["test_time_seconds"] = test_time
    typer.echo(metrics)

    # Save model to Hugging Face Hub and metrics locally
    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    project_name = os.getenv("PROJECT_NAME", "text-class-tutorial")
    hf_profile = os.getenv("HF_PROFILE_NAME", "user")
    repo_name = f"{hf_profile}/{project_name}-setfit"
    typer.echo(f"Pushing model to Hugging Face Hub as: {repo_name}")

    # Push model to hub
    model.push_to_hub(repo_name)
    typer.echo(f"Model successfully pushed to hub: {repo_name}")

    # Save metrics locally only
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{datetime_str}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    typer.echo(f"Metrics saved locally to: {metrics_path}")

    # Generate predictions and print details
    texts = test_dataset["text"]
    y_true = test_dataset["label"]
    y_pred = model.predict(texts)
    print_details(y_pred, y_true, SAMPLE_LABELS)

    # Example prediction
    test_example = test_dataset[0]["text"]
    typer.echo(test_example)
    typer.echo(model.predict(test_example))
    typer.echo(model.predict_proba(test_example))


if __name__ == "__main__":
    app()
