from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction
import numpy as np

CLF_METRICS = {
    "accuracy": accuracy_score,
}

def compute_clf_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    metrics = {}
    for metric_name, metric_fn in CLF_METRICS.items():
        metrics[metric_name] = metric_fn(p.label_ids, preds)
    return metrics

def get_eval_fn(data_cfg: str):
    if data_cfg['eval_metric'] == "classification":
        return compute_clf_metrics
    else:
        raise ValueError(f"Unsupported task type: {data_cfg['eval_metric']}")