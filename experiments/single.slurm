#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=1-00:00:00
#SBATCH --job-name=fax1k
#SBATCH --output=/home/s2233827/slurm/%x.out

# move into project and load wandb env vars
cd $HOME/fax
set -a; source .env; set +a

# run training
uv run fax/training/train.py \
    --matchup "XvX" \
    --mds "$HOME/Data/fax/mds" \
    --n-epochs 1000 \
    --n-finetune-epochs 100 \
    --wandb
