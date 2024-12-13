import torch
from typing import Dict
import os

def get_lasted_checkpoint(args, run_name: str) -> str:
    checkpoint_dir = os.path.join(args.output_dir, run_name)
    checkpoints = [f.path for f in os.scandir(checkpoint_dir) if f.is_dir() and 'checkpoint' in f.path]
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)  # Get the most recent checkpoint
    return None
  
def calculate_modal_loss(outputs: Dict[str, torch.Tensor], loss_type: str = 'mse') -> torch.Tensor:
    loss = {}
    total_loss = 0.
    target = outputs['target']
    for modal in ['optical', 'radar', 'multi']:
        if f'{modal}_recon' in outputs:
            recon = outputs[f'{modal}_recon'] # B, C, HW, D
            channel_mask = outputs[f'{modal}_channel_mask'] # B, C
            pos_mask = outputs[f'{modal}_pos_mask'] # B, HW
            
            if loss_type == 'mse':
                patch_error = torch.mean((recon - target) ** 2, dim=-1) # B, C, HW
            else:
                patch_error = torch.mean((recon - target).abs(), dim=-1) # B, C, HW
                
            optical_error, radar_error = patch_error[:, :-2], patch_error[:, -2:]
            optical_channel_mask, radar_channel_mask = channel_mask[:, :-2], channel_mask[:, -2:]
            
            pos_loss, channel_loss = [], []
            
            # for each modal
            for modal_error, modal_channel_mask in zip([optical_error, radar_error], [optical_channel_mask, radar_channel_mask]):
                modal_pos_loss = torch.mean(modal_error, dim=1) # B, HW
                modal_channel_loss = torch.mean(modal_error, dim=2) # B, C
                    
                if pos_mask.sum() != 0:
                    pos_loss.append((modal_pos_loss * pos_mask).sum() / pos_mask.sum())
                    
                if modal_channel_mask.sum() != 0:
                    channel_loss.append((modal_channel_loss * modal_channel_mask).sum() / modal_channel_mask.sum())
            assert len(pos_loss) > 0 and len(channel_loss) > 0

            pos_loss = sum(pos_loss) / len(pos_loss)
            channel_loss = sum(channel_loss) / len(channel_loss)
            
            loss[f"{modal}_pos_loss"] = pos_loss
            loss[f"{modal}_channel_loss"] = channel_loss
            loss[f"{modal}_loss"] = pos_loss + channel_loss
            total_loss += loss[f"{modal}_loss"]
    loss['total_loss'] = total_loss
    return loss['total_loss']