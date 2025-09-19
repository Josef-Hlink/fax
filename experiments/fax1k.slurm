#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00  # one day
#SBATCH --job-name=fax1k
#SBATCH --array=0-3  # 4 jobs in the sweep
#SBATCH --output=/home/s2233827/slurm/%x_%A/%a.out


# move into project and load wandb env vars
cd $HOME/fax/
set -a; source .env; set +a

# create slurm output dir
mkdir -p "$HOME/slurm/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"

# sweep values
MATCHUPS=(XvX XvF FvX FvF)

uv run fax/training/train.py \
    --matchup "${MATCHUPS[$SLURM_ARRAY_TASK_ID]}" \
    --mds "$HOME/Data/fax/mds${SLURM_ARRAY_TASK_ID}" \
    --n-epochs 1000 \
    --n-finetune-epochs 100 \
    --wandb
