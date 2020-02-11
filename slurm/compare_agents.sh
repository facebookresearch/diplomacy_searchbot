#!/bin/bash
#
# Example:
# NUM_TRIALS=2 slurm/compare_agents.sh mila mila /checkpoint/alerer/fairdiplomacy/compare_test1

set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 agent_one agent_six output_dir [script_args]"
fi

LAUNCH="$(dirname $0)/launch_slurm.sh"

# it's more efficient to do this once in advance
export CODE_CHECKPOINT="$($(dirname $0)/checkpoint_repo.sh)"
export SCRIPT="bin/compare_agents.py"


NUM_TRIALS=${NUM_TRIALS:-10}
AGENT_ONE=$1
AGENT_SIX=$2
export CHECKPOINT_BASE=$3
# shift off the first 3 arguments; any additional args get passed to the script
shift 3

mkdir -p $CHECKPOINT_BASE

for POWER in AUSTRIA ENGLAND FRANCE GERMANY ITALY RUSSIA TURKEY; do
    for SEED in $(seq $NUM_TRIALS); do
        NAME="compare_${POWER}_${SEED}" $LAUNCH $AGENT_ONE $AGENT_SIX $POWER --seed $SEED $@
    done
done
