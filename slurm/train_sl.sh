#!/bin/bash
set -e

LAUNCH="$(dirname $0)/launch_slurm.sh"
DEFAULT_ARGS="--data-cache /checkpoint/jsgray/diplomacy/data_cache.pth"

# it's more efficient to do this once in advance
export CODE_CHECKPOINT="$($(dirname $0)/checkpoint_repo.sh)"

# if you want to e.g. run on dev partition, you can do
# export PARTITION=dev

NAME="sl_b1000" $LAUNCH $DEFAULT_ARGS --batch-size 1000
