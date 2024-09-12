import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GeospatialFM Training Arguments")

    # Dataset arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the SSL4EO dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")

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

    # Decoder arguments
    parser.add_argument("--decoder_embed_dim", type=int, default=512, help="Embedding dimension for decoder")
    parser.add_argument("--decoder_depth", type=int, default=8, help="Number of transformer layers for decoder")
    parser.add_argument("--decoder_num_heads", type=int, default=16, help="Number of attention heads for decoder")
    parser.add_argument("--decoder_out_chans", type=int, default=1, help="Number of output channels for decoder")
    parser.add_argument("--decoder_out_dims", type=int, default=1, help="Number of output dimensions for decoder")

    # extra model arguments
    parser.add_argument("--return_dict", type=bool, default=False, help="Return a dictionary instead of a tuple")
    parser.add_argument("--norm_pix_loss", type=bool, default=True, help="Use normalized pixel loss")
    parser.add_argument("--use_perception_field_mask", type=bool, default=False, help="Use perception field mask")
    parser.add_argument("--attention_radius", type=int, default=640, help="Attention radius for perception field mask")

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Mask ratio for MAE")
    parser.add_argument("--channel_mask_ratio", type=float, default=0.5, help="Channel mask ratio for MAE")

    # Logging and checkpointing arguments
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for saving logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saving model checkpoints")
    parser.add_argument("--save_frequency", type=int, default=10, help="Frequency of saving checkpoints (in epochs)")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cuda or cpu)")

    return parser.parse_args()
