from transformers import Trainer
import numpy as np
from typing import Dict

class MAETrainer(Trainer):
    def __init__(self, modal_mode=None, **kwargs):
        super().__init__(**kwargs)
        self.modal_mode = modal_mode

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.modal_mode == "random":
            modal = np.random.choice(['multi', 'optical', 'radar'])
        else:
            modal = self.modal_mode
            
        outputs = model(**inputs, modal = modal)
        
        assert self.compute_loss_func is not None, "compute_loss_func is not set"
        loss = self.compute_loss_func(outputs)

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
