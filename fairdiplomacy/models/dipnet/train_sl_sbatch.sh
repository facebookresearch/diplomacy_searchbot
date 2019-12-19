#!/bin/bash
#SBATCH --job-name=train_sl
#SBATCH --partition=learnfair
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=0
#SBATCH --time=2880
#SBATCH --requeue


CHECKPOINT_DIR=/checkpoint/$USER/jobs/$SLURM_JOB_ID
mkdir -p $CHECKPOINT_DIR

cat <<ENDNOTE >$CHECKPOINT_DIR/note.txt
actual season/power emb
fixed loc_idxs, with fixed adj, fixed tf masking
fixed incompatible builds masking
ENDNOTE

srun --label \
    python train_sl.py \
    --data-dir /private/home/jsgray/code/fairdiplomacy/fairdiplomacy/data/mila_dataset/data \
    --checkpoint $CHECKPOINT_DIR/checkpoint.pth \
    --batch-size 600 \
    --lr 0.001 \
    --num-dataloader-workers 10 \
    --teacher-force 1.0 \
    --lstm-dropout 0 \
    --validate-every 1000 \
    1>$CHECKPOINT_DIR/stdout.log \
    2>$CHECKPOINT_DIR/stderr.log
