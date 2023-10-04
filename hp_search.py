from transformers import TrainingArguments, Trainer
from torchgeo.samplers import RandomGeoSampler
import wandb

from GeospatialFM.utils import setup, get_eval_fn, get_args_parser
from GeospatialFM.data import *
from GeospatialFM.models import *

args = get_args_parser().parse_args()

def train():
    # Initialize wandb run
    cfg, run = setup(args)    

    training_args = TrainingArguments(**cfg['TRAINER'])
    training_args.learning_rate = run.config.learning_rate
    model = get_model(cfg['MODEL'])
    train_ds, val_ds, test_ds = get_datasets(cfg['DATASET'])
    compute_metrics = get_eval_fn(cfg['DATASET'])

    trainer = Trainer(
        model= model,                # the instantiated 🤗 Transformers model to be trained
        args=training_args,                   # training arguments, defined above
        train_dataset=train_ds,    # training dataset
        eval_dataset=val_ds,      # evaluation dataset
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()

    # Log metrics to wandb
    wandb.log(metrics)

    # End the run
    run.finish()

if __name__ == '__main__':
    sweep_id = "wnoul4nv"  # Replace with your SWEEP_ID from step 3
    wandb.agent(sweep_id, train)