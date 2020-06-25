#!/bin/bash
set -e

cd "$($(dirname $0)/checkpoint_repo.sh)"

export HOURS=48
export NAME=${NAME:-my_model}
PARTITION=${PARTITION:-learnfair}

python run.py \
    --cfg conf/c02_sup_train/sl.prototxt \
    I.launcher=slurm_8gpus \
    launcher.slurm.hours=$HOURS \
    launcher.slurm.partition=$PARTITION \
    num_epochs=200 \
    --exp_id_pattern_override /checkpoint/$USER/fairdiplomacy/$NAME $@
