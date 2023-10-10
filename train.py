from transformers import TrainingArguments, Trainer
from torchgeo.samplers import RandomGeoSampler

from GeospatialFM.utils import setup, get_eval_fn, get_args_parser, get_call_back
from GeospatialFM.data import *
from GeospatialFM.models import *

from torch.utils.data import ConcatDataset

args = get_args_parser().parse_args()
cfg, _ = setup(args)

training_args = TrainingArguments(**cfg['TRAINER'])
model = construct_model(cfg['MODEL'])
train_ds, val_ds, test_ds = get_datasets(cfg['DATASET'])
compute_metrics = get_eval_fn(cfg['DATASET'])
call_back = get_call_back(cfg['TRAINER_EXTRA'])

trainer = Trainer(
    model=model,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_ds,    # training dataset
    eval_dataset=test_ds,      # evaluation dataset
    compute_metrics=compute_metrics,
    callbacks=call_back,
)

trainer.train()
results = trainer.evaluate(eval_dataset=test_ds)
# output the evaluation results to cfg['TRAINER']['logging_dir']
print("Test Set Results:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
trainer.save_model()