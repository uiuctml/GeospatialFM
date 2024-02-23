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
        micro_mAP = average_precision_score(targets.cpu().numpy(), outputs.cpu().numpy(), average='micro')
        macro_mAP = average_precision_score(targets.cpu().numpy(), outputs.cpu().numpy(), average='macro')
        return micro_mAP, macro_mAP

    def get_metrics(self):
        micro_mAP, macro_mAP = self.compute()
        return {"micro_mAP": micro_mAP, "macro_mAP": macro_mAP}

class F1Meter(MetricTracker):
    def __init__(self):
        super().__init__()
        self.outputs = []
        self.targets = []

    def update(self, output, target):
        self.outputs.append(output.flatten(1))
        self.targets.append(target.flatten(1))

    def compute(self, average='micro'):
        outputs = torch.sigmoid(torch.cat(self.outputs, dim=0))
        targets = torch.cat(self.targets, dim=0).to(torch.float32)
        preds = (outputs >= 0.5).to(torch.float32)
        precision = precision_score(targets.cpu().numpy(), preds.cpu().numpy(), average=average, zero_division=0)
        recall = recall_score(targets.cpu().numpy(), preds.cpu().numpy(), average=average, zero_division=0)
        f1 = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average=average, zero_division=0)
        return precision, recall, f1

    def get_metrics(self):
        precision, recall, f1 = self.compute()
        return {"precision": precision, "recall": recall, "f1": f1}

def get_eval_meter(eval_metric):
    if eval_metric == "accuracy":
        return AccuracyMeter()
    elif eval_metric == "mAP":
        return mAPMeter()
    elif eval_metric == "f1":
        return F1Meter()
    else:
        raise ValueError(f"Unsupported task type: {eval_metric}")