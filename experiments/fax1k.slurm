#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00  # one day
#SBATCH --job-name=fax1k
#SBATCH --array=4-7  # 4 jobs total
#SBATCH --output=/home/s2233827/slurm/%x_%A/%a.out

# move into project and load wandb env vars
cd $HOME/fax/
set -a; source .env; set +a

# create slurm stdout dir
mkdir -p "$HOME/slurm/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"

# matchup mapping
MATCHUPS=(XvX XvF FvX FvF XvX XvF FvX FvF)
MATCHUP="${MATCHUPS[$SLURM_ARRAY_TASK_ID]}"

# data mapping
DATA="$HOME/Data/fax/mds${SLURM_ARRAY_TASK_ID}"

# run training
uv run fax/training/train.py \
    --matchup "$MATCHUP" \
    --mds "$DATA" \
    --n-epochs 1000 \
    --n-finetune-epochs 100 \
    --wandb
