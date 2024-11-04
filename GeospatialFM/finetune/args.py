import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="GeospatialFM Finetune Arguments")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--task_type", type=str, choices=["classification", "multilabel", "segmentation"], required=True, help="Task type")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the GFMBench")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of subprocesses to use for data loading")
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Whether to pin memory for data loading")
    parser.add_argument("--use_8bit", action="store_true", help="Whether to use 8-bit data loading")
    parser.add_argument("--modal", type=str, default="optical", choices=["optical", "radar", "multi"], help="Modal to finetune on")
    parser.add_argument("--crop_size", type=int, default=None, help="Crop size for training")
    parser.add_argument("--scale", type=float, default=None, help="Scale for training")
    parser.add_argument("--random_rotation", action="store_true", help="Whether to use random rotation for training")

    # Model arguments
    parser.add_argument("--patch_size", type=int, default=16, help="Size of patches for hyperspectral patch embedding")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--channel_embed_dims_per_head", type=int, default=4, help="Number of channel embedding dimensions per head")
    parser.add_argument("--depth", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP ratio")
    parser.add_argument("--qkv_bias", type=bool, default=True, help="Use bias in qkv projections")
    parser.add_argument("--qk_norm", type=bool, default=False, help="Use qk normalization")
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="Drop path rate")
    parser.add_argument("--drop_path_uniform", type=bool, default=False, help="Use uniform drop path")
    parser.add_argument("--init_values", type=float, default=None, help="Init values for LayerScale")
    parser.add_argument("--attn_drop", type=float, default=0.0, help="Attention dropout rate")
    parser.add_argument("--proj_drop", type=float, default=0.0, help="Projection dropout rate")

    # extra model arguments
    parser.add_argument("--return_dict", action="store_true", help="Return a dictionary instead of a tuple")
    parser.add_argument("--use_perception_field_mask", action="store_true", help="Use perception field mask")
    parser.add_argument("--attention_radius", type=int, default=320, help="Attention radius for perception field mask")

    # Training arguments
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam optimizer beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam optimizer beta2")
    parser.add_argument("--adam_weight_decay", type=float, default=0.05, help="Adam optimizer weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam optimizer epsilon")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Type of learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.2, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether to use gradient checkpointing to save memory at the expense of slower backward pass")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm for gradient clipping")
    parser.add_argument("--early_stop_steps", type=int, default=None, help="Stop training after X steps. Used for debugging.")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze the encoder")
    
    # Evaluation arguments
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X updates steps")
    parser.add_argument("--eval_strategy", type=str, choices=["epoch", "steps", "no"], default="epoch", help="Evaluation strategy")
    
    # Logging and saving arguments
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model checkpoints and logs")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--report_to", type=str, default="wandb", help="Where to report results to (tensorboard, wandb, etc.)")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=None, help="If set, deletes the older checkpoints in output_dir")
    parser.add_argument("--wandb_dir", type=str, default="wandb", help="Directory to save wandb logs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=[None, "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to the pretrained model")
    
    # Append run name to directories
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.logging_dir = os.path.join(args.logging_dir, args.run_name)
    return args
