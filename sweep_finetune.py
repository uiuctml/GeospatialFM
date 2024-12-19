import argparse
import os
from dataclasses import dataclass
from typing import List, Optional
import subprocess
import random

@dataclass
class DatasetConfig:
    name: str
    task_type: str
    crop_size: int
    batch_size: int = 32
    grad_accum_steps: int = 2
    num_epochs: int = 10
    early_stopping_patience: int = 3
    train_frac: Optional[float] = None
    val_frac: Optional[float] = None

# Dataset-specific configurations
DATASET_CONFIGS = {
    "bigearthnet": DatasetConfig(
        name="bigearthnet",
        task_type="multilabel",
        crop_size=112,
        train_frac=0.1,
        val_frac=0.1,
    ),
    "dfc2020": DatasetConfig(
        name="dfc2020",
        task_type="segmentation",
        crop_size=96,
    ),
    "segmunich": DatasetConfig(
        name="segmunich",
        task_type="segmentation",
        crop_size=128,
        grad_accum_steps=4,
        batch_size=16,
    ),
    "eurosat": DatasetConfig(
        name="eurosat",
        task_type="classification",
        crop_size=64,
        batch_size=64,
        grad_accum_steps=1,
        num_epochs=20,
        early_stopping_patience=5,
    ),
    "so2sat": DatasetConfig(
        name="so2sat",
        task_type="classification",
        crop_size=32,
        batch_size=64,
        grad_accum_steps=1,
        num_epochs=20,
        early_stopping_patience=5,
        train_frac=0.1,
        val_frac=0.1,
    ),
    "marida": DatasetConfig(
        name="marida",
        task_type="segmentation",
        crop_size=96,
    ),
}

def generate_finetune_command(
    root_dir: str,
    run_name: str,
    dataset_config: DatasetConfig,
    embed_dims: int,
    depth: int,
    learning_rate: str,
    port: int,
    checkpoint: int = 24600,
    moe: int = 0,
    scale: int = 1,
    attention_radius: int = 640,
    topk: int = 3,
) -> str:
    cmd = [
        "accelerate launch",
        f"--main_process_port {port}",
        "GeospatialFM/finetune/finetune.py",
        f"--data_dir {root_dir}/data/geospatial-2/",
        f"--dataset_name {dataset_config.name}",
        f"--task_type {dataset_config.task_type}",
        f"--scale {scale}",
        "--modal optical",
        "--return_dict",
        f"--per_device_train_batch_size {dataset_config.batch_size}",
        f"--gradient_accumulation_steps {dataset_config.grad_accum_steps}",
        f"--num_train_epochs {dataset_config.num_epochs}",
        f"--learning_rate {learning_rate}",
        "--weight_decay 0.01",
        "--warmup_steps 0",
        "--warmup_ratio 0.2",
        "--report_to none",
        "--save_total_limit 1",
        "--seed 42",
        "--mixed_precision bf16",
        "--dataloader_num_workers 32",
        "--dataloader_pin_memory",
        f"--output_dir {root_dir}/results/models",
        f"--logging_dir {root_dir}/results/logs",
        f"--wandb_dir {root_dir}/results/",
        f"--run_name {run_name}",
        "--lr_scheduler_type cosine",
        f"--channel_embed_dims_per_head {embed_dims}",
        "--use_perception_field_mask",
        f"--pretrained_model_path {root_dir}/results/models/LESSVIT_b{embed_dims}_d{depth}/checkpoint-{checkpoint}/model.safetensors",
        f"--attention_radius {attention_radius}",
        "--use_early_stopping",
        f"--early_stopping_patience {dataset_config.early_stopping_patience}",
        f"--crop_size {dataset_config.crop_size}",
        "--init_values 1",
    ]

    if moe > 0:
        cmd.append("--use_moe")
        cmd.append(f"--num_experts {moe}")
        cmd.append(f"--topk {topk}")
    if dataset_config.train_frac:
        cmd.append(f"--train_frac {dataset_config.train_frac}")
    if dataset_config.val_frac:
        cmd.append(f"--val_frac {dataset_config.val_frac}")

    return " \\\n    ".join(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--root_dir", default="/home/haozhesi/GeospatialFM")
    parser.add_argument("--gpu_devices", default="0,1,2,3")
    args = parser.parse_args()

    # Set environment variables
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{args.root_dir}"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    dataset_config = DATASET_CONFIGS[args.dataset]
    
    # sweep fields
    learning_rates = ["3e-5", "5e-5", "8e-5", "1e-4", "3e-4", "5e-4", "8e-4", "1e-3"]
    embed_dims_list = [2]  # Modify as needed
    depth_list = [4, 8]    # Modify as needed
    # adjustable parameters
    moe = 0
    scale = 1
    
    # random port
    port = random.randint(10000, 65535)

    for embed_dims in embed_dims_list:
        for depth in depth_list:
            for lr in learning_rates:
                run_name = f"LESSVIT_b{embed_dims}_d{depth}_{dataset_config.name}_lr{lr}_scale{scale}_moe{moe}"
                cmd = generate_finetune_command(
                    root_dir=args.root_dir,
                    dataset_config=dataset_config,
                    embed_dims=embed_dims,
                    depth=depth,
                    learning_rate=lr,
                    port=port,
                    run_name=run_name,
                    # adjustable parameters
                    moe=moe,
                    scale=scale,
                )
                
                print(f"Running command:\n{cmd}")
                subprocess.run(cmd, shell=True)
                
                # save the command to a file
                with open(f"{args.root_dir}/results/models/{dataset_config.name}/{run_name}/launch_finetune.sh", "w") as f:
                    f.write(cmd)

if __name__ == "__main__":
    main()