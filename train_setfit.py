import os
import json
from datetime import datetime, timezone
from typing import Optional

import typer
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

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
    train_dataset = Dataset.from_parquet("data/train.parquet")
    eval_dataset = Dataset.from_parquet("data/eval.parquet")
    test_dataset = Dataset.from_parquet("data/test.parquet")

    if prefix is not None:

        def add_prefix(example):
            return {"text": f"{prefix}{example['text']}"}

        train_dataset = train_dataset.map(add_prefix)
        eval_dataset = eval_dataset.map(add_prefix)
        test_dataset = test_dataset.map(add_prefix)

    print("Datasets loaded")

    # Load a SetFit model from Hub with the provided model name
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
        },  # Map dataset columns to text/label expected by trainer
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate(test_dataset)
    typer.echo(metrics)

    # Saving the trained model in a folder that depends on the model name
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_path = f"{OUTPUT_DIR}/{model_name}_{datetime_str}"
    model.save_pretrained(save_path)

    texts = test_dataset["text"]
    y_true = test_dataset["label"]
    y_pred = model.predict(texts)
    print_details(y_pred, y_true, SAMPLE_LABELS)

    with open(os.path.join(save_path, f"metrics_{datetime_str}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    test_example = test_dataset[0]["text"]
    typer.echo(test_example)
    typer.echo(model.predict(test_example))
    typer.echo(model.predict_proba(test_example))


if __name__ == "__main__":
    app()
