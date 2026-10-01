# Chronos-2 pretraining

Pretrains [Chronos-2](https://arxiv.org/abs/2510.15821) from random initialization on multiple GPUs and nodes.
The model, loss and batch format come from the official
[`chronos-forecasting`](https://github.com/amazon-science/chronos-forecasting) package, which ships fine-tuning
code but no pretraining script. This directory adds the missing pieces: a data pipeline that mixes many sources
and a `DistributedDataParallel` training loop.

> **Status:** not run yet. The code has only been syntax-checked, so start with the smoke test below.

## Files

- `train.py`: training loop (DDP, bf16 autocast, TF32, fused AdamW, warmup + cosine schedule, resumable checkpoints)
- `data.py`: data sources and batch construction
- `configs/base.yaml`: Chronos-2 base (120M parameters), `configs/debug.yaml`: tiny model on toy data
- `scripts/train.slurm`: multi-node launcher

## Running

```bash
pip install -r requirements.txt
```

Smoke test, on a single GPU or on CPU:

```bash
python train.py --config configs/debug.yaml
```

One node with 4 GPUs:

```bash
torchrun --nproc-per-node=4 train.py --config configs/base.yaml
```

Multiple nodes with SLURM (edit the `#SBATCH` lines first):

```bash
sbatch scripts/train.slurm configs/base.yaml
```

Launching the same command again resumes from the latest checkpoint in `output_dir`. Checkpoints load with the
official pipeline:

```python
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("runs/chronos2-base/checkpoint-200000")
```

## Data

Downloading and preparing the data is not part of this directory. `data.sources` in the config lists where the
training tasks come from, and each source is drawn with a probability proportional to its `weight`:

- `type: arrow` reads real series through HF `datasets`. `path` is a directory written by `Dataset.save_to_disk`
  or a glob of `.parquet` / `.arrow` files, and `column` holds one series per row (1-d, or 2-d with shape
  `(n_variates, length)`). Prefer `save_to_disk` directories on a multi-node run: parquet files are converted
  to Arrow in the HF cache the first time they are opened.
- `type: callable` calls `fn(rng, length, **kwargs)` of your own, given as `"module:function"`, for synthetic data
  generated on the fly. It returns a `data.Task`: an array of shape `(n_variates, length)` with rows ordered
  `[targets | past-only covariates | known-future covariates]`, plus `n_targets` and `n_future_covariates`.

According to the authors, the multivariate and covariate abilities of Chronos-2 come entirely from synthetic
tasks, so the `callable` sources matter as much as the real data.

## What is and is not from the paper

- From the released model: the architecture in `configs/base.yaml`, the quantile loss and the batch format.
- From the paper: two stages (context length 2048 with few output patches, then 8192 with more), and the
  number of output patches being sampled randomly for each batch.
- Not disclosed by the authors, so the values here are guesses to tune: learning rate and schedule, weight decay,
  number of steps and their split between the stages, batch sizes, the maximum number of output patches of
  each stage, and the data mixture. The learning rate, the weight decay and the 200k steps are borrowed from
  the original Chronos paper.
- The batch size of the second stage has not been profiled. Run a few steps of a config containing only that
  stage before committing to a long run.
- There is no validation loop. Evaluate checkpoints with the official pipeline.
