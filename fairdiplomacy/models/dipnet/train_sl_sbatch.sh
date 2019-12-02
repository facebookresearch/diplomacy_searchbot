#!/bin/bash
#SBATCH --job-name=train_sl
#SBATCH --partition=learnfair
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=1440
#SBATCH --requeue


CHECKPOINT_DIR=/checkpoint/$USER/jobs/$SLURM_JOB_ID
mkdir -p $CHECKPOINT_DIR

cat <<ENDNOTE >$CHECKPOINT_DIR/note.txt
mila encoding
small data
ENDNOTE

srun --label \
    python train_sl.py \
    --data-dir /private/home/jsgray/code/fairdiplomacy/fairdiplomacy/data/mila_dataset/small_data \
    --checkpoint $CHECKPOINT_DIR/checkpoint.pth \
    --batch-size 280 \
    --lr 0.05 \
    --num-dataloader-workers 10 \
    1>$CHECKPOINT_DIR/stdout.log \
    2>$CHECKPOINT_DIR/stderr.log
