from transformers import TrainingArguments, Trainer

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