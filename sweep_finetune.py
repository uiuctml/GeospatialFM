import argparse
import os
from dataclasses import dataclass
from typing import List, Optional
import subprocess
import random
import shutil
import torch
import time

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
    dataset_version: Optional[str] = None
    
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
        batch_size=32,
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
        num_epochs=10,
        early_stopping_patience=5,
        batch_size=16,
        dataset_version = "etm_oli_toa_nlcd"
    ),
}

@dataclass
class ModelConfig:
    embed_dims: int
    depth: int
    num_heads: int

MODEL_CONFIGS = {
    "LESSVIT-S": ModelConfig(embed_dims=384, depth=12, num_heads=6),
    "LESSVIT-B": ModelConfig(embed_dims=768, depth=12, num_heads=12),
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
    modal: str = "optical",
    dataset_version: Optional[str] = None,
    use_optuna: bool = False,
    rank: int = 1,
    model: str = "LESSVIT-B",
) -> str:
    script = "finetune.py"
    dataset_config.batch_size = per_device_batch_size if per_device_batch_size else dataset_config.batch_size
    batch_size = 1024 if linear_probe else dataset_config.batch_size
    grad_accum_steps = 1 if linear_probe else dataset_config.effective_batch_size // n_gpus // dataset_config.batch_size
    num_epochs = 100 if linear_probe else dataset_config.num_epochs
    dataset_version = dataset_config.dataset_version if not dataset_version else dataset_version
    model_config = MODEL_CONFIGS[model.upper()]
    
    if "lessvit-s" in model.lower():
        prefix = "s"
    elif "lessvit-b" in model.lower():
        prefix = "b"
    else:
        raise ValueError(f"Invalid model: {model}")
    
    cmd = [
        "accelerate launch",
        f"--main_process_port {port}"
    ]
    if accelerator_config:
        cmd.append(accelerator_config)
        
    model_name = f"LESSVIT_{prefix}{embed_dims}_d{depth}_r{rank}"
    
    cmd.extend([
        f"GeospatialFM/finetune/{script}",
        f"--data_dir {root_dir}/data/geospatial-2/",
        f"--dataset_name {dataset_config.name}",
        f"--task_type {dataset_config.task_type}",
        f"--scale {scale}",
        f"--modal {modal}",
        "--return_dict",
        f"--embed_dim {model_config.embed_dims}",
        f"--depth {model_config.depth}",
        f"--num_heads {model_config.num_heads}",
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
        f"--pretrained_model_path {root_dir}/results/models/{model_name}/checkpoint-{checkpoint}/model.safetensors",
        f"--attention_radius {attention_radius}",
        f"--crop_size {dataset_config.crop_size}",
        "--init_values 1",
        f"--rank {rank}",
    ])
    
    if linear_probe:
        cmd.append("--lp")
        cmd.append("--freeze_encoder")

    if regenerate_embeddings:
        cmd.append("--regenerate_embeddings")
    
    if not linear_probe:
        cmd.append("--use_early_stopping")
        cmd.append(f"--early_stopping_patience {dataset_config.early_stopping_patience}")
    else:
        cmd.append("--save_strategy no")
    
    if dataset_version:
        cmd.append(f"--dataset_version {dataset_version}")

    if moe > 0:
        cmd.append("--use_moe")
        cmd.append(f"--num_experts {moe}")
        cmd.append(f"--topk {topk}")
    if dataset_config.train_frac:
        cmd.append(f"--train_frac {dataset_config.train_frac}")
    if dataset_config.val_frac:
        cmd.append(f"--val_frac {dataset_config.val_frac}")
        
    if use_optuna:
        cmd.append("--use_optuna")

    return " \\\n    ".join(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--dataset_version", "-v", default=None, type=str, help="Data version")
    parser.add_argument("--root_dir", default="/home/haozhesi/GeospatialFM")
    parser.add_argument("--gpu_devices", "-d", default="0,1,2,3")
    parser.add_argument("--lp", action="store_true", help="Run in linear probe mode")
    parser.add_argument("--moe", default=0, type=int, help="Number of experts")
    parser.add_argument("--regenerate_embeddings", action="store_true", help="Regenerate embeddings")
    parser.add_argument("--checkpoint", default=24600, type=int, help="Checkpoint to load")
    parser.add_argument("--per_device_batch_size", "-b", default=None, type=int, help="Per device batch size")
    parser.add_argument("--scale", default=None, type=int, help="Scale of the model")
    parser.add_argument("--topk", default=3, type=int, help="Topk for MoE")
    parser.add_argument("--modal", default="optical", type=str, help="Modal to finetune")
    parser.add_argument("--attention_radius", default=640, type=int, help="Attention radius for perception field mask")
    parser.add_argument("--use_optuna", action="store_true", help="Use Optuna to find the best hyper-parameters")
    parser.add_argument("--rank", default=1, type=int, help="Rank of the model")
    parser.add_argument("--model", default="LESSVIT-S", type=str, help="Model to use")
    # reproduce hyper-parameters
    parser.add_argument("--lr", default=None, type=float, help="Override learning rate")
    
    args = parser.parse_args()

    # Set environment variables
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{args.root_dir}"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    dataset_config = DATASET_CONFIGS[args.dataset]
    accelerator_config = "--config_file ~/.cache/huggingface/accelerate/single_gpu_config.yaml"
    
    # sweep fields
    if args.lp:
        learning_rates = ["5e-3", "8e-3", "1e-2", "3e-2", "5e-2", "8e-2", "1e-1", "3e-1"]
    else:
        learning_rates = ["3e-5", "5e-5", "8e-5", "1e-4", "3e-4", "5e-4", "8e-4", "1e-3"]
        
    if args.lr:
        learning_rates = [args.lr]
    
    embed_dims_list = [2]  # Modify as needed
    depth_list = [4]    # Modify as needed
    rank_list = [1, 2, 4, 8]

    # adjustable parameters
    moe = args.moe
    scale = args.scale if args.scale else dataset_config.scale
    # random port
    port = random.randint(10000, 65535)
    command_list = []

    regenerate_embeddings = args.regenerate_embeddings
    for lr in learning_rates:
        # loop over all the combinations of the model parameters
        for embed_dims in embed_dims_list:
            for depth in depth_list:
                for rank in rank_list:
                    run_name = f"{args.model}{embed_dims}_d{depth}_r{rank}_{dataset_config.name}_lr{lr}_scale{scale}"
                    if args.dataset_version:
                        run_name += f"_{args.dataset_version}"
                    if args.moe > 0:
                        run_name += f"_moe{args.moe}"
                    if args.topk != 3:
                        run_name += f"_topk{args.topk}"
                    if args.lp:
                        run_name += "_lp"
                    if args.modal != "optical":
                        run_name += f"_{args.modal}"
                    if args.checkpoint != 24600:
                        run_name += f"_ckpt{args.checkpoint}"    
                    if args.attention_radius != 640:
                        run_name += f"_ar{args.attention_radius}"
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
                        n_gpus=1,
                        per_device_batch_size=args.per_device_batch_size,
                        topk=args.topk,
                        # adjustable parameters
                        moe=moe,
                        scale=scale,
                        linear_probe=args.lp,
                        accelerator_config=accelerator_config,
                        regenerate_embeddings=regenerate_embeddings,
                        modal=args.modal,
                        dataset_version=args.dataset_version,
                        attention_radius=args.attention_radius,
                        rank=args.rank,
                        model=args.model,
                    )
                    
                    command_list.append(cmd)
                    
                    # save the command to a file
                    # create the directory if it doesn't exist
                    os.makedirs(f"{args.root_dir}/results/models/{dataset_config.name}/{run_name}", exist_ok=True)
                    with open(f"{args.root_dir}/results/models/{dataset_config.name}/{run_name}/launch_finetune.sh", "w") as f:
                        f.write(cmd)
        regenerate_embeddings = False
                        
    # run the commands in parallel
    if len(args.gpu_devices.split(",")) > 1:
        multi_gpu_launcher(command_list)
    else:
        local_launcher(command_list)

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                print(f"Running command:\n{cmd}")
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    main()