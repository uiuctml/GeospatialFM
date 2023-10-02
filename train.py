from transformers import TrainingArguments, Trainer
from torchgeo.samplers import RandomGeoSampler

from GeospatialFM.utils import setup, get_eval_fn, get_args_parser
from GeospatialFM.data import *
from GeospatialFM.models import *

args = get_args_parser().parse_args()
cfg = setup(args)

training_args = TrainingArguments(**cfg['TRAINER'])
model = get_model(cfg['MODEL'])
train_ds, val_ds, test_ds = get_datasets(cfg['DATASET'])
compute_metrics = get_eval_fn(cfg['DATASET'])

trainer = Trainer(
    model=model,                # the instantiated 🤗 Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=train_ds,    # training dataset
    eval_dataset=val_ds,      # evaluation dataset
    compute_metrics=compute_metrics,
)

trainer.train()