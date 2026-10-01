"""
Pretrain Chronos-2 from scratch on one or more GPUs with DistributedDataParallel.

    python train.py --config configs/debug.yaml                        # single GPU (or CPU)
    torchrun --nproc-per-node=4 train.py --config configs/base.yaml    # one node, 4 GPUs
    sbatch scripts/train.slurm                                         # multiple nodes

Checkpoints are written with `save_pretrained`, so they can be loaded with `Chronos2Pipeline.from_pretrained`.
Launching the same command again resumes from the latest checkpoint in the output directory.
"""

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from chronos.chronos2 import Chronos2CoreConfig, Chronos2Model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from data import PretrainingDataset


def setup_distributed() -> tuple[int, int, torch.device]:
    """Joins the process group described by the environment variables set by torchrun, if there is one."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    return rank, world_size, device


def build_model(model_cfg: dict) -> Chronos2Model:
    config = Chronos2CoreConfig(
        **model_cfg["core"],
        chronos_config=dict(model_cfg["chronos"]),
        chronos_pipeline_class="Chronos2Pipeline",
    )
    return Chronos2Model(config)


def build_optimizer(model: Chronos2Model, train_cfg: dict, device: torch.device) -> torch.optim.Optimizer:
    # no weight decay for biases and layer norm weights
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(train_cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(train_cfg["learning_rate"]),
        fused=device.type == "cuda",
    )


def learning_rate_at(step: int, train_cfg: dict, total_steps: int) -> float:
    """Linear warmup followed by cosine decay to zero over all the stages."""
    peak, warmup_steps = float(train_cfg["learning_rate"]), int(train_cfg["warmup_steps"])
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_dataloader(
    model: Chronos2Model, data_cfg: dict, stage: dict, seed: list[int], pin_memory: bool
) -> DataLoader:
    assert stage["context_length"] <= model.chronos_config.context_length
    assert stage["max_output_patches"] <= model.chronos_config.max_output_patches

    dataset = PretrainingDataset(
        sources=data_cfg["sources"],
        context_length=stage["context_length"],
        batch_size=stage["batch_size"],
        patch_size=model.chronos_config.output_patch_size,
        max_output_patches=stage["max_output_patches"],
        min_past=data_cfg["min_past"],
        max_group_size=data_cfg["max_group_size"],
        seed=seed,
    )
    # batch_size=None because the dataset directly returns batches instead of individual elements
    return DataLoader(dataset, batch_size=None, num_workers=data_cfg["num_workers"], pin_memory=pin_memory)


def list_checkpoints(output_dir: Path) -> list[Path]:
    return sorted(output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))


def save_checkpoint(
    model: Chronos2Model, optimizer: torch.optim.Optimizer, step: int, output_dir: Path, keep_last: int
) -> None:
    # The checkpoint is written to a temporary directory and then renamed, so that a job which is killed
    # while saving never leaves a broken checkpoint behind for the next run to resume from.
    tmp_dir = output_dir / f"tmp-checkpoint-{step}"
    final_dir = output_dir / f"checkpoint-{step}"
    model.save_pretrained(tmp_dir)
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, tmp_dir / "training_state.pt")
    shutil.rmtree(final_dir, ignore_errors=True)
    tmp_dir.rename(final_dir)

    if keep_last > 0:
        for stale_dir in list_checkpoints(output_dir)[:-keep_last]:
            shutil.rmtree(stale_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain Chronos-2 from scratch.")
    parser.add_argument("--config", required=True, help="Path to a YAML config, see configs/")
    parser.add_argument("--output-dir", default=None, help="Overrides output_dir of the config")
    parser.add_argument(
        "--resume",
        default="auto",
        help='"auto" resumes from the latest checkpoint in the output directory, if there is one. '
        '"none" starts from scratch. Anything else is taken as the path to a checkpoint directory.',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_cfg, stages = cfg["train"], cfg["stages"]
    output_dir = Path(args.output_dir or cfg["output_dir"])
    total_steps = sum(stage["steps"] for stage in stages)

    rank, world_size, device = setup_distributed()
    is_main = rank == 0
    torch.set_float32_matmul_precision("high")  # allows TF32 matmuls

    if args.resume == "auto":
        checkpoints = list_checkpoints(output_dir)
        resume_dir = checkpoints[-1] if len(checkpoints) > 0 else None
    else:
        resume_dir = None if args.resume == "none" else Path(args.resume)

    # every rank builds the same initial model, the seeds only diverge afterwards (e.g., for dropout)
    torch.manual_seed(train_cfg["seed"])
    model = build_model(cfg["model"]) if resume_dir is None else Chronos2Model.from_pretrained(resume_dir)
    model.to(device)
    torch.manual_seed(train_cfg["seed"] + rank)

    optimizer = build_optimizer(model, train_cfg, device)
    step = 0
    if resume_dir is not None:
        training_state = torch.load(resume_dir / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(training_state["optimizer"])
        step = training_state["step"]

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Chronos-2 with {n_params / 1e6:.1f}M parameters on {world_size} x {device.type}", flush=True)
        if resume_dir is not None:
            print(f"Resumed from {resume_dir} at step {step}", flush=True)

    ddp_model: torch.nn.Module = model
    if world_size > 1:
        ddp_model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    ddp_model.train()

    use_bf16 = bool(train_cfg["bf16"]) and device.type == "cuda"
    loss_sum = torch.zeros((), device=device)
    n_steps = n_series = n_skipped = 0
    start_time = time.perf_counter()

    stage_end = 0
    for stage in stages:
        stage_end += stage["steps"]
        if step >= stage_end:
            continue
        if is_main:
            print(f"Training up to step {stage_end} with {stage}", flush=True)

        # the step is part of the seed, so that a resumed run does not replay the batches it has already seen
        dataloader = build_dataloader(
            model, cfg["data"], stage, seed=[train_cfg["seed"], rank, step], pin_memory=device.type == "cuda"
        )
        for batch in dataloader:
            lr = learning_rate_at(step, train_cfg, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = ddp_model(**batch).loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
            if torch.isfinite(grad_norm):
                optimizer.step()
                loss_sum += loss.detach().float()
                n_steps += 1
            else:
                # Gradients are averaged over all ranks during the backward pass, so a non-finite loss
                # on any rank makes every rank skip this update.
                n_skipped += 1
            n_series += len(batch["context"])
            step += 1

            if step % train_cfg["log_every"] == 0:
                mean_loss = loss_sum / max(n_steps, 1)
                if world_size > 1:
                    dist.all_reduce(mean_loss)
                    mean_loss /= world_size
                if is_main:
                    metrics = {
                        "step": step,
                        "loss": mean_loss.item(),
                        "learning_rate": lr,
                        "grad_norm": grad_norm.item(),
                        "series_per_second": world_size * n_series / (time.perf_counter() - start_time),
                        "skipped_steps": n_skipped,
                    }
                    print(
                        f"step {step}/{total_steps} | loss {metrics['loss']:.4f} | lr {lr:.2e} | "
                        f"grad norm {metrics['grad_norm']:.2f} | {metrics['series_per_second']:.0f} series/s",
                        flush=True,
                    )
                    with open(output_dir / "metrics.jsonl", "a") as f:
                        f.write(json.dumps(metrics) + "\n")
                loss_sum.zero_()
                n_steps = n_series = 0
                start_time = time.perf_counter()

            if is_main and (step % train_cfg["save_every"] == 0 or step == total_steps):
                save_checkpoint(model, optimizer, step, output_dir, train_cfg["keep_last"])

            if step >= stage_end:
                break
        del dataloader

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
