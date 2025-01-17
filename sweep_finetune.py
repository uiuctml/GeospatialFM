import argparse
import os
from dataclasses import dataclass
from typing import List, Optional
import subprocess
import random
import shutil
@dataclass
class DatasetConfig:
    name: str
    task_type: str
    crop_size: int
    batch_size: int = 32
    effective_batch_size: int = 256
    num_epochs: int = 10
    early_stopping_patience: int = 3
    train_frac: Optional[float] = None
    val_frac: Optional[float] = None
    scale: int = 1

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
        batch_size=16,
    ),
    "eurosat": DatasetConfig(
        name="eurosat",
        task_type="classification",
        crop_size=64,
        batch_size=64,
        num_epochs=20,
        early_stopping_patience=5,
        scale=2
    ),
    "so2sat": DatasetConfig(
        name="so2sat",
        task_type="classification",
        crop_size=32,
        batch_size=64,
        num_epochs=20,
        early_stopping_patience=5,
        train_frac=0.1,
        val_frac=0.1,
        scale=4
    ),
    "marida": DatasetConfig(
        name="marida",
        task_type="segmentation",
        crop_size=96,
    ),
    "landsat": DatasetConfig(
        name="landsat",
        task_type="segmentation",
        crop_size=128,
        batch_size=16,
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
    linear_probe: bool = False,
    accelerator_config: str = "",
    regenerate_embeddings: bool = False,
    n_gpus: int = 4,
    per_device_batch_size: Optional[int] = None,
) -> str:
    script = "linear_probe.py" if linear_probe else "finetune.py"
    dataset_config.batch_size = per_device_batch_size if per_device_batch_size else dataset_config.batch_size
    batch_size = 1024 if linear_probe else dataset_config.batch_size
    grad_accum_steps = 1 if linear_probe else dataset_config.effective_batch_size // n_gpus // dataset_config.batch_size
    num_epochs = 100 if linear_probe else dataset_config.num_epochs
    
    cmd = [
        "accelerate launch",
        f"--main_process_port {port}"
    ]
    if accelerator_config:
        cmd.append(accelerator_config)
    
    cmd.extend([
        f"GeospatialFM/finetune/{script}",
        f"--data_dir {root_dir}/data/geospatial-2/",
        f"--dataset_name {dataset_config.name}",
        f"--task_type {dataset_config.task_type}",
        f"--scale {dataset_config.scale}",
        "--modal optical",
        "--return_dict",
        f"--per_device_train_batch_size {batch_size}",
        f"--gradient_accumulation_steps {grad_accum_steps}",
        f"--num_train_epochs {num_epochs}",
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
        f"--crop_size {dataset_config.crop_size}",
        "--init_values 1",
    ])
    
    if regenerate_embeddings:
        cmd.append("--regenerate_embeddings")
    
    if not linear_probe:
        cmd.append("--use_early_stopping")
        cmd.append(f"--early_stopping_patience {dataset_config.early_stopping_patience}")
    else:
        cmd.append("--save_strategy no")

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
    parser.add_argument("--lp", action="store_true", help="Run in linear probe mode")
    parser.add_argument("--moe", default=0, type=int, help="Number of experts")
    parser.add_argument("--regenerate_embeddings", action="store_true", help="Regenerate embeddings")
    parser.add_argument("--checkpoint", default=24600, type=int, help="Checkpoint to load")
    parser.add_argument("--per_device_batch_size", "-b", default=None, type=int, help="Per device batch size")
    parser.add_argument("--scale", default=None, type=int, help="Scale of the model")
    # reproduce hyper-parameters
    parser.add_argument("--lr", default=None, type=float, help="Override learning rate")
    
    args = parser.parse_args()

    # Set environment variables
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{args.root_dir}"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    dataset_config = DATASET_CONFIGS[args.dataset]
    accelerator_config = "--config_file ~/.cache/huggingface/accelerate/lp_config.yaml" if (args.lp or len(args.gpu_devices.split(",")) == 1) else ""
    
    # sweep fields
    if args.lp:
        learning_rates = ["5e-3", "8e-3", "1e-2", "3e-2", "5e-2", "8e-2", "1e-1", "3e-1"]
    else:
        learning_rates = ["3e-5", "5e-5", "8e-5", "1e-4", "3e-4", "5e-4", "8e-4", "1e-3"]
        
    if args.lr:
        learning_rates = [args.lr]
    
    embed_dims_list = [2]  # Modify as needed
    depth_list = [4]    # Modify as needed
    # adjustable parameters
    moe = args.moe
    scale = args.scale if args.scale else dataset_config.scale
    # random port
    port = random.randint(10000, 65535)

    for embed_dims in embed_dims_list:
        for depth in depth_list:
            regenerate_embeddings = args.regenerate_embeddings
            for lr in learning_rates:
                run_name = f"LESSVIT_b{embed_dims}_d{depth}_{dataset_config.name}_lr{lr}_scale{scale}_moe{moe}"
                if args.lp:
                    run_name += "_lp"
                if args.checkpoint != 24600:
                    run_name += f"_ckpt{args.checkpoint}"
                # check if the run_name already exists and completed
                if os.path.exists(f"{args.root_dir}/results/models/{dataset_config.name}/{run_name}/test_results.json"):
                    if args.regenerate_embeddings:
                        print(f"Redo the experiment for {run_name}")
                        shutil.rmtree(f"{args.root_dir}/results/models/{dataset_config.name}/{run_name}")
                    else:
                        print(f"Run {run_name} already exists and completed")
                        continue
                    
                cmd = generate_finetune_command(
                    root_dir=args.root_dir,
                    dataset_config=dataset_config,
                    embed_dims=embed_dims,
                    depth=depth,
                    learning_rate=lr,
                    port=port,
                    run_name=run_name,
                    checkpoint=args.checkpoint,
                    n_gpus=len(args.gpu_devices.split(",")),
                    per_device_batch_size=args.per_device_batch_size,
                    # adjustable parameters
                    moe=moe,
                    scale=scale,
                    linear_probe=args.lp,
                    accelerator_config=accelerator_config,
                    regenerate_embeddings=regenerate_embeddings,
                )
                
                print(f"Running command:\n{cmd}")
                subprocess.run(cmd, shell=True)
                
                regenerate_embeddings = False
                
                # save the command to a file
                with open(f"{args.root_dir}/results/models/{dataset_config.name}/{run_name}/launch_finetune.sh", "w") as f:
                    f.write(cmd)

if __name__ == "__main__":
    main()