#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00  # one day
#SBATCH --job-name=s3-fax1k
#SBATCH --array=0-7  # 8 jobs total
#SBATCH --output=/home/s2233827/slurm/%x_%A/%a.out

# move into project and load wandb env vars
cd $HOME/fax/
set -a; source .env; set +a

# create slurm stdout dir
mkdir -p "$HOME/slurm/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"

# create s3 data dir
DATA="$HOME/Data/s3/mds${SLURM_ARRAY_TASK_ID}"
mkdir -p "$DATA"

# matchup mapping
MATCHUPS=(XvX XvF FvX FvF XvX XvF FvX FvF)
MATCHUP="${MATCHUPS[$SLURM_ARRAY_TASK_ID]}"

# bucket mapping
BUCKETS=(nofox onefox onefox twofox nofox onefox onefox twofox)
BUCKET="${BUCKETS[$SLURM_ARRAY_TASK_ID]}"

echo "Job $SLURM_ARRAY_TASK_ID using matchup=$MATCHUP bucket=$BUCKET"

echo "Copying training data into $DATA/$BUCKET ..."
aws s3 cp --recursive s3://fax-ssbm/$BUCKET/ "$DATA/$BUCKET/"

# if FvF, don’t copy twice — train + finetune both use the same bucket
if [[ "$MATCHUP" == "FvF" ]]; then
    echo "FvF: skipping duplicate finetune copy, reuse $BUCKET ..."
else
    echo "Copying finetune data into $DATA/twofox ..."
    aws s3 cp --recursive s3://fax-ssbm/twofox/ "$DATA/twofox/"
fi

# run training
uv run fax/training/train.py \
    --matchup "$MATCHUP" \
    --mds "$DATA" \
    --n-epochs 1000 \
    --n-finetune-epochs 10 \
    --wandb

# clean up data
echo "Cleaning up $DATA ..."
rm -rf $DATA