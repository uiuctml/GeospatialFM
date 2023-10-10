from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import EarlyStoppingCallback

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