#!/bin/bash
set -e

cd "$($(dirname $0)/checkpoint_repo.sh)"

export HOURS=48
export NAME="my_model"

python run.py \
    --cfg conf/c02_sup_train/sl.prototxt \
    I.launcher=slurm_8gpus \
    launcher.slurm.hours=$HOURS \
    num_epochs=200 \
    --exp_id_pattern_override /checkpoint/$USER/fairdiplomacy/$NAME
