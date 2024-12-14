from transformers import Trainer
import numpy as np
from typing import Dict

class MAETrainer(Trainer):
    def __init__(self, modal_mode=None, **kwargs):
        super().__init__(**kwargs)
        self.modal_mode = modal_mode

    def compute_loss(self, model, inputs, return_outputs=False, mask_ratio=None, channel_mask_ratio=None):
        optical = inputs.get("optical").to(self.accelerator.device, dtype=self.weight_dtype)
        radar = inputs.get("radar").to(self.accelerator.device, dtype=self.weight_dtype)
        optical_channel_wv = inputs.get("optical_channel_wv")
        radar_channel_wv = inputs.get("radar_channel_wv")
        spatial_resolution = inputs.get("spatial_resolution")
        
        if self.modal_mode == "random":
            modal = np.random.choice(['multi', 'optical', 'radar'])
        else:
            modal = self.modal_mode
        
        outputs = model(
            optical=optical,
            radar=radar,
            optical_channel_wv=optical_channel_wv,
            radar_channel_wv=radar_channel_wv,
            mask_ratio=mask_ratio,
            channel_mask_ratio=channel_mask_ratio,
            spatial_resolution=spatial_resolution,
            modal=modal
        )

        loss = self.calculate_modal_loss(outputs, self.loss_type)

        return (loss, outputs) if return_outputs else loss

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
