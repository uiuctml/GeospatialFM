from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import EarlyStoppingCallback
import torch.nn as nn
from typing import Dict, Union, Any
import torch
import math

def get_call_back(cfg):
    call_back_list = []
    if cfg['early_stop']['enable']:
        early_stop_cfg = cfg['early_stop']
        early_stop = EarlyStoppingCallback(early_stopping_patience=early_stop_cfg['patience'], early_stopping_threshold=early_stop_cfg['threshold'])
        call_back_list.append(early_stop)

    if len(call_back_list) == 0:
        return None
    return call_back_list

class GeoTrainer(Trainer):
    # TODO: modify the trainer to support custom sampler for geospaital data
    pass
    # def __init__(self, 
    #              *args,
    #              train_sampler=None, 
    #              eval_sampler=None,
    #              **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.train_sampler = train_sampler
    #     self.eval_sampler = eval_sampler

class CropTrainer(Trainer):
    def __init__(self, loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        #     labels = None
        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))
        model_out = model(inputs['image'], inputs['radar'])

        losses = self.loss(**model_out, output_dict=True)

        loss = sum(losses.values())

        # if labels is not None:
        #     loss = self.label_smoother(outputs, labels)
        # else:
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, model_out) if return_outputs else loss