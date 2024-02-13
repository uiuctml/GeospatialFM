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


class MetricTracker(object):
    def __init__(self):
        self.count = 0
    
    def update(self, output, target):
        pass

    def compute(self):
        pass

    def get_metrics(self):
        pass


class AccuracyMeter(MetricTracker):
    def __init__(self):
        super().__init__()
        self.correct = 0

    def update(self, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        self.correct += pred.eq(target.view_as(pred)).sum().item()
        self.count += target.size(0)

    def compute(self):
        return self.correct / self.count

    def get_metrics(self):
        return {"accuracy": self.compute()}


class mAPMeter(MetricTracker):
    def __init__(self):
        super().__init__()
        self.outputs = []
        self.targets = []

    def update(self, output, target):
        self.outputs.append(output)
        self.targets.append(target)

    def compute(self):
        outputs = torch.cat(self.outputs, dim=0).float()
        targets = torch.cat(self.targets, dim=0).float()
        return average_precision_score(targets.cpu().numpy(), outputs.cpu().numpy(), average='micro')

    def get_metrics(self):
        return {"mAP": self.compute()}


class F1Meter(MetricTracker):
    def __init__(self):
        super().__init__()
        self.outputs = []
        self.targets = []

    def update(self, output, target):
        self.outputs.append(output)
        self.targets.append(target)

    def compute(self):
        outputs = torch.sigmoid(torch.cat(self.outputs, dim=0))
        targets = torch.cat(self.targets, dim=0)
        preds = (outputs >= 0.5).to(torch.float32)
        precision = precision_score(targets.cpu().numpy(), preds.cpu().numpy(), average='sample')
        recall = recall_score(targets.cpu().numpy(), preds.cpu().numpy(), average='sample')
        f1 = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='sample')
        return precision, recall, f1

    def get_metrics(self):
        precision, recall, f1 = self.compute()
        return {"precision": precision, "recall": recall, "f1": f1}


# def cal_acc(model_out, labek, return_dict=True):
#     metrics = {}
#     preds = torch.argmax(model_out, dim=1)
#     image_acc = (preds == label).sum().item() / len(label)
#     metrics['accuracy'] = image_acc
#     if return_dict:
#         return metrics
#     else:
#         return metrics["accuracy"]

# def cal_mAP(model_out, label, return_dict=True):
#     metrics = {}
#     preds = torch.sigmoid(model_out).detach().cpu().float().numpy()
#     label = label.detach().cpu().float().numpy()
#     image_mAP = average_precision_score(label, preds, average='micro')
#     metrics['mAP'] = image_mAP
#     if return_dict:
#         return metrics
#     else:
#         return metrics["mAP"]

# def cal_f1(model_out, label, return_dict=True):
#     metrics = {}
#     model_out = F.sigmoid(model_out)
#     preds = (model_out >= 0.5).to(torch.float32)
#     preds = preds.flatten(1).detach().cpu().numpy()
#     label = label.flatten(1).to(torch.float32).detach().cpu().numpy()
#     # TP = (preds * label).sum(dim=1)
#     # FN = ((preds == 0) * (label == 1)).sum(dim=1)
#     # FP = ((preds == 1) * (label == 0)).sum(dim=1)
#     # recall_all = TP / (TP + FN)
#     # precision_all = TP / (TP + FP)
#     # recall_all[torch.isnan(recall_all)] = 0
#     # precision_all[torch.isnan(precision_all)] = 0
#     # f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all)
#     # f1_all[torch.isnan(f1_all)] = 0
#     # f1 = torch.mean(f1_all)
#     # precision = torch.mean(precision_all)
#     # recall = torch.mean(recall_all)

#     precision = precision_score(label, preds, average='sample')
#     recall = recall_score(label, preds, average='sample')
#     f1 = f1_score(label, preds, average='sample')
#     metrics['f1'] = f1
#     metrics['precision'] = precision
#     metrics['recall'] = recall
#     if return_dict:
#         return metrics
#     else:
#         return metrics["precision"], metrics["recall"], metrics["f1"]

def get_eval_meter(eval_metric):
    if eval_metric == "accuracy":
        return AccuracyMeter()
    elif eval_metric == "mAP":
        return mAPMeter()
    elif eval_metric == "f1":
        return F1Meter()
    else:
        raise ValueError(f"Unsupported task type: {eval_metric}")