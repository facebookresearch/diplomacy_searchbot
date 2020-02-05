#!/bin/bash
set -e

LAUNCH="$(dirname $0)/launch_slurm.sh"

DEFAULT_ARGS="--data-dir /private/home/jsgray/code/fairdiplomacy/fairdiplomacy/data/mila_dataset/data \
--cache-dir /private/home/alerer/git/fairdiplomacy/local/data_cache.pt \
--batch-size 1000"

# it's more efficient to do this once in advance
export CODE_CHECKPOINT="$($(dirname $0)/checkpoint_repo.sh)"

# if you want to e.g. run on dev partition, you can do
# export PARTITION=dev

NAME="sl6_lr1e-3" $LAUNCH $DEFAULT_ARGS --lr 1e-3
NAME="sl6_lr1e-4" $LAUNCH $DEFAULT_ARGS --lr 1e-4
