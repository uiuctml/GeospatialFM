from GeospatialFM.models.downstream_models import LESSWithProjectionConfig, LESSWithUPerNetConfig, LESSWithProjection, LESSWithUPerNet
import torch.nn as nn
import torch
from functools import partial
from typing import Dict
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, average_precision_score, jaccard_score
from GeospatialFM.models.baseline_models.downstream_baseline_models import BaselineWithProjectionConfig, BaselineWithUPerNetConfig, BaselineWithProjection, BaselineWithUPerNet

def get_task_model(args, num_classes=None, image_size=None):
    if args.task_type == "classification" or args.task_type == "multilabel":
        assert num_classes is not None
        config = LESSWithProjectionConfig(num_labels=num_classes, **vars(args))
        model = LESSWithProjection(config)
    elif args.task_type == "segmentation":
        assert num_classes is not None and image_size is not None
        config = LESSWithUPerNetConfig(num_labels=num_classes, image_size=image_size, **vars(args))
        model = LESSWithUPerNet(config)
    else:
        raise NotImplementedError
    return model

def get_baseline_model(args, num_classes=None, image_size=None):
    if args.task_type == "classification" or args.task_type == "multilabel":
        assert num_classes is not None
        config = BaselineWithProjectionConfig(num_labels=num_classes, **vars(args))
        model = BaselineWithProjection(config)
    elif args.task_type == "segmentation" or args.task_type == "landsat":
        assert num_classes is not None and image_size is not None
        config = BaselineWithUPerNetConfig(num_labels=num_classes, image_size=image_size, **vars(args))
        model = BaselineWithUPerNet(config)
    else:
        raise NotImplementedError
    return model

def load_ckpt(args):
    pass

def custom_loss_function(outputs, labels, num_items_in_batch, loss_fct):
    """
    Custom loss function.
    Modify this function based on your specific task.
    """
    logits = outputs.get("logits")
    # loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss_fct(logits.to(torch.float32), labels.to(torch.long))
    return loss

def get_loss_fn(task_type):
    if task_type == "classification" or task_type == "segmentation" or task_type == "landsat":
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255)
    elif task_type == "multilabel":
        loss_fct = torch.nn.MultiLabelSoftMarginLoss()
    else:
        raise NotImplementedError
    
    loss_fn = partial(custom_loss_function, loss_fct=loss_fct)
    return loss_fn

def compute_metrics_acc(eval_pred: EvalPrediction) -> Dict:
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())

    return {"accuracy": accuracy}

def compute_metrics_mAP(eval_pred: EvalPrediction) -> Dict:
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    macro_mAP = average_precision_score(labels, predictions, average="macro")
    micro_mAP = average_precision_score(labels, predictions, average="micro")

    # return {"macro_mAP": macro_mAP, "micro_mAP": micro_mAP}
    return {"micro_mAP": micro_mAP}

def compute_metrics_IoU(eval_pred: EvalPrediction, num_classes=11) -> Dict:
    # predictions = eval_pred.predictions
    # labels = eval_pred.label_ids

    # predictions = np.argmax(predictions, axis=1)
    # IoU = jaccard_score(labels.flatten(), predictions.flatten(), average="macro")
    predictions = torch.tensor(eval_pred.predictions)
    labels = torch.tensor(eval_pred.label_ids)
    predictions = torch.argmax(predictions, dim=1)
    predictions = predictions.flatten()
    labels = labels.flatten()

    n = num_classes
    mat = torch.zeros((n, n), dtype=torch.int64)
    with torch.no_grad():
        k = (labels >= 0) & (labels < n)
        inds = n * labels[k].to(torch.int64) + predictions[k]
        mat += torch.bincount(inds, minlength=n**2).reshape(n, n)
    mat_to_float = mat.to(torch.float32)
    iu = torch.diag(mat_to_float) / (mat_to_float.sum(dim=1) + mat_to_float.sum(dim=0) - torch.diag(mat_to_float))
    iu[torch.isnan(iu)] = 0.0
    IoU = torch.mean(iu).item()

    return {"IoU": IoU}

def get_metric(task_type, num_classes=None):
    if task_type == "classification":
        return compute_metrics_acc, "accuracy"
    elif task_type == "multilabel":
        return compute_metrics_mAP, "micro_mAP"
    elif task_type == "segmentation" or task_type == "landsat":
        return partial(compute_metrics_IoU, num_classes=num_classes), "IoU"
    else:
        raise NotImplementedError