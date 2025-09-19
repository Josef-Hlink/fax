#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --constraint="Geforce.rtx2080Ti|A100.4g.40gb|A100.3g.40gb"
#SBATCH --cpus-per-task=10
#SBATCH --mem=16GB
#SBATCH --time=4:00
#SBATCH --job-name=faxtest
#SBATCH --output=slurm-%j.out

cd $HOME/fax
set -a; source .env; set +a

uv run fax/training/train.py \
    --matchup "FvF" \
    --n-epochs 100 \
    --n-finetune-epochs 10 \
    --wandb
