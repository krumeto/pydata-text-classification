from typing import Sequence, Dict, Any, Union, Optional

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, multilabel_confusion_matrix
)

from settings import SAMPLE_LABELS

def compute_metrics(
    y_pred: Sequence[int],
    y_test: Sequence[int],
) -> Dict[str, Any]:
    # ensure list of ints
    y_pred = list(y_pred)
    y_test = list(y_test)
    assert len(y_pred) == len(y_test), "y_pred/y_test length mismatch"

    # full report
    report = classification_report(
        y_test,
        y_pred,
        target_names=SAMPLE_LABELS,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        'accuracy_report': report['accuracy'],
        'precision_weighted': report['weighted avg']['precision'],
        'recall_weighted':    report['weighted avg']['recall'],
        'f1_weighted':        report['weighted avg']['f1-score'],
    }

    for label in SAMPLE_LABELS:
        cls = report[label]
        metrics[f'{label}.precision'] = cls['precision']
        metrics[f'{label}.recall']    = cls['recall']
        metrics[f'{label}.f1_score']  = cls['f1-score']
        metrics[f'{label}.support']   = int(cls['support'])

    return metrics


def print_details(
    y_pred: Sequence[int],
    y_test: Sequence[int],
    labels: Sequence[str]
) -> None:
    """
    Print detailed classification report and confusion matrices.
    """
    print(classification_report(y_test, y_pred, target_names=labels))
    cms = multilabel_confusion_matrix(y_test, y_pred)
    for label, cm in zip(labels, cms):
        print(f'Confusion matrix for {label}:\n{cm}\n')