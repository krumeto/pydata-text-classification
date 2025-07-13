import os
import json
from datetime import datetime, timezone

import typer
from datasets import load_dataset, Dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, multilabel_confusion_matrix

app = typer.Typer()

OUTPUT_DIR = 'models/setfit'

SAMPLE_LABELS = ['GENERATING COMMUNICATIVE TEXT',
'INFORMATION SEARCH',
'SOFTWARE DEVELOPMENT',
'GENERATING CREATIVE TEXT',
'HOMEWORK PROBLEM']

def compute_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    # Compute per-class metrics
    report = classification_report(
        y_test,
        y_pred,
        target_names=SAMPLE_LABELS,
        output_dict=True,
        zero_division=0
    )
    print(report)
    # Initialize metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }

    confusion_matrices = multilabel_confusion_matrix(y_test, y_pred)
    for idx, label in enumerate(SAMPLE_LABELS):
        print(f"Confusion matrix for {label}:")
        print(confusion_matrices[idx])

    # Add per-class metrics to the metrics dictionary
    for label in SAMPLE_LABELS:
        metrics[f"{label}_precision"] = report[label]['precision']
        metrics[f"{label}_recall"] = report[label]['recall']
        metrics[f"{label}_f1_score"] = report[label]['f1-score']
        metrics[f"{label}_support"] = report[label]['support']

    return metrics

@app.command()
def main(model_name: str):
    train_dataset = Dataset.from_parquet("data/train.parquet")
    eval_dataset = Dataset.from_parquet("data/eval.parquet")
    test_dataset = Dataset.from_parquet("data/test.parquet")

    print('Datasets loaded')

    # Load a SetFit model from Hub with the provided model name
    model = SetFitModel.from_pretrained(
        f"{model_name}",
        labels=SAMPLE_LABELS,
    )
    print("Model loaded")
    model.model_body.max_seq_length = 512

    print(f"{model_name} successfully initialized!")
    args = TrainingArguments(
        batch_size=16,
#        max_steps=70,
        logging_steps=5,
        num_epochs=1,
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
        column_mapping={"text": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate(test_dataset)
    print(metrics)

    # Saving the trained model in a folder that depends on the model name
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datetime_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_path = f"{OUTPUT_DIR}/{model_name}_{datetime_str}"
    model.save_pretrained(save_path)

    with open(os.path.join(save_path, f"metrics_{datetime_str}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    test_example = test_dataset[0]['text']
    print(test_example)
    print(model.predict(test_example))
    print(model.predict_proba(test_example))

if __name__ == "__main__":
    app()