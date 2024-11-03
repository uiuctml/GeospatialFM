from GeospatialFM.models.downstream_models import LESSWithProjectionConfig, LESSWithUPerNetConfig, LESSWithProjection, LESSWithUPerNet
import torch.nn as nn
import torch
from functools import partial
from typing import Dict
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, average_precision_score, jaccard_score

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

def custom_loss_function(outputs, labels, num_items_in_batch, loss_fct):
    """
    Custom loss function.
    Modify this function based on your specific task.
    """
    logits = outputs.get("logits")
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

def get_loss_fn(task_type):
    if task_type == "classification" or task_type == "segmentation":
        loss_fct = torch.nn.CrossEntropyLoss()
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

    return {"macro_mAP": macro_mAP, "micro_mAP": micro_mAP}

def compute_metrics_IoU(eval_pred: EvalPrediction) -> Dict:
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    predictions = np.argmax(predictions, axis=1)
    IoU = jaccard_score(labels.flatten(), predictions.flatten(), average="macro")

    return {"IoU": IoU}

def get_metric(task_type):
    if task_type == "classification":
        return compute_metrics_acc
    elif task_type == "multilabel":
        return compute_metrics_mAP
    elif task_type == "segmentation":
        return compute_metrics_IoU
    else:
        raise NotImplementedError