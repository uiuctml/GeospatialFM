from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
import numpy as np
import torch
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_acc(model_out, labek, return_dict=True):
    metrics = {}
    preds = torch.argmax(model_out, dim=1)
    image_acc = (preds == label).sum().item() / len(label)
    metrics['accuracy'] = image_acc
    if return_dict:
        return metrics
    else:
        return metrics["accuracy"]

def cal_mAP(model_out, label, return_dict=True):
    metrics = {}
    preds = torch.sigmoid(model_out).detach().cpu().float().numpy()
    label = label.detach().cpu().float().numpy()
    image_mAP = average_precision_score(label, preds, average='samples')
    metrics['mAP'] = image_mAP
    if return_dict:
        return metrics
    else:
        return metrics["mAP"]

def cal_f1(model_out, label, return_dict=True):
    metrics = {}
    model_out = F.sigmoid(model_out)
    preds = (model_out >= 0.5).to(torch.float32)
    preds = preds.flatten(1)
    label = label.flatten(1).to(torch.float32)
    TP = (preds * label).sum(dim=1)
    FN = ((preds == 0) * (label == 1)).sum(dim=1)
    FP = ((preds == 1) * (label == 0)).sum(dim=1)
    recall_all = TP / (TP + FN)
    precision_all = TP / (TP + FP)
    recall_all[torch.isnan(recall_all)] = 0
    precision_all[torch.isnan(precision_all)] = 0
    f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all)
    f1_all[torch.isnan(f1_all)] = 0
    f1 = torch.mean(f1_all)
    precision = torch.mean(precision_all)
    recall = torch.mean(recall_all)

    # precision = precision_score(label, preds, average='macro')
    # recall = recall_score(label, preds, average='macro')
    # image_f1 = f1_score(label, preds, average='macro')
    metrics['f1'] = f1
    metrics['precision'] = precision
    metrics['recall'] = recall
    if return_dict:
        return metrics
    else:
        return metrics["precision"], metrics["recall"], metrics["f1"]

def get_eval_fn(eval_metric):
    if eval_metric == "accuracy":
        return cal_acc
    elif eval_metric == "mAP":
        return cal_mAP
    elif eval_metric == "f1":
        return cal_f1
    else:
        raise ValueError(f"Unsupported task type: {eval_metric}")