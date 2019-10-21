#!/bin/bash
#SBATCH --job-name=train_sl
#SBATCH --partition=learnfair
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=720


CHECKPOINT_DIR=/checkpoint/$USER/jobs/$SLURM_JOB_ID
mkdir -p $CHECKPOINT_DIR


srun --label \
    python train_sl.py \
    --data-dir /private/home/jsgray/code/fairdiplomacy/fairdiplomacy/data/out \
    --checkpoint $CHECKPOINT_DIR/checkpoint.pth \
    1>$CHECKPOINT_DIR/stdout.log \
    2>$CHECKPOINT_DIR/stderr.log
