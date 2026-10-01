#!/bin/bash
#SBATCH --job-name=chronos2-pretrain
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1   # one torchrun agent per node, which starts one training process per GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=40    # at least gpus-per-node * (data.num_workers + 1)
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

# Submit from the chronos2_pretrain directory:
#   sbatch scripts/train.slurm [config]
# Submitting the same job again resumes from the latest checkpoint in the output directory.

set -euo pipefail

CONFIG=${1:-configs/base.yaml}
GPUS_PER_NODE=4  # keep in sync with --gpus-per-node

# source /path/to/venv/bin/activate

# The batch script runs on the first node of the allocation, which hosts the rendezvous for all agents.
# The port is derived from the job id, so that two jobs sharing a head node do not collide.
RDZV_ENDPOINT="$(hostname):$((20000 + SLURM_JOB_ID % 20000))"

srun --cpus-per-task="$SLURM_CPUS_PER_TASK" torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$RDZV_ENDPOINT" \
    train.py --config "$CONFIG"
