import os
import json
from datetime import datetime, timezone
import typer
import numpy as np
from time import perf_counter
from datasets import Dataset
from model2vec.train import StaticModelForClassification
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from typing import Sequence, Dict, Any, Union, Optional

from settings import SAMPLE_LABELS, MAIN_OUTPUT_DIR
from utils import compute_metrics, print_details

OUTPUT_DIR = MAIN_OUTPUT_DIR + "model2vec"

app = typer.Typer()

@app.command()
def main(
    model_name: str = typer.Argument(
        ..., help='Pre-trained model2vec checkpoint name (e.g., minishlab/potion-base-32m)'
    ),
    prefix: Optional[str] = typer.Option(
        None,
        '--prefix', '-p',
        help="Optionally prepend this string to every example's text before training."
    ),
):
    """
    Train a model2vec classifier and compute metrics aligned with the SetFit baseline.
    """
    train_ds = Dataset.from_parquet('data/train.parquet')
    eval_ds = Dataset.from_parquet('data/eval.parquet')
    test_ds = Dataset.from_parquet('data/test.parquet')

    if prefix:
        def add_prefix(ex):
            ex['text'] = prefix + ex['text']
            return ex
        train_ds = train_ds.map(add_prefix)
        eval_ds = eval_ds.map(add_prefix)
        test_ds = test_ds.map(add_prefix)

    typer.echo('Datasets loaded')

    typer.echo(f'Loading model2vec classifier {model_name}')
    classifier = StaticModelForClassification.from_pretrained(model_name=model_name)
    typer.echo('Classifier loaded')

    texts = train_ds['text']
    labels = train_ds['label']
    typer.echo('Starting training...')
    start = perf_counter()
    classifier = classifier.fit(texts, labels)
    elapsed = int(perf_counter() - start)
    typer.echo(f'Training took {elapsed} seconds.')

    typer.echo('Evaluating on test set...')
    y_pred_test = classifier.predict(test_ds['text'])
    y_true_test = test_ds['label']
    test_metrics = compute_metrics(y_pred_test, y_true_test)
    typer.echo('Test metrics:')
    typer.echo(test_metrics)

    print_details(y_pred_test, y_true_test, SAMPLE_LABELS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dt_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(OUTPUT_DIR, f'{model_name.replace("/", "_")}_{dt_str}')
    os.makedirs(save_path, exist_ok=True)
    typer.echo(f'Saving model to {save_path}')
    pipeline = classifier.to_pipeline()
    pipeline.save_pretrained(save_path)

    metrics_path = os.path.join(save_path, f'metrics_{dt_str}.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    typer.echo(f'Metrics saved to {metrics_path}')

    # Example prediction
    test_example = test_ds[0]['text']
    typer.echo(f'Example text: {test_example}')
    pred = pipeline.predict([test_example])[0]
    proba = pipeline.predict_proba([test_example])[0]
    typer.echo(f'Predicted class: {pred}')
    typer.echo(f'Predicted probabilities: {proba}')

if __name__ == '__main__':
    app()
