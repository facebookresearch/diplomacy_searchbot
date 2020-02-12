#!/bin/bash
#
# Example:
# NUM_TRIALS=2 slurm/compare_agents.sh mila mila compare_test1

set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 agent_one agent_six name [script_args]"
  exit 1
fi

LAUNCH="$(dirname $0)/launch_slurm.sh"

# it's more efficient to do this once in advance
export CODE_CHECKPOINT="$($(dirname $0)/checkpoint_repo.sh)"
export SCRIPT="bin/compare_agents.py"

export GPU=${GPU:-1}
export CPU=${CPU:-10}

NUM_TRIALS=${NUM_TRIALS:-10}
AGENT_ONE=$1
AGENT_SIX=$2
NAME=$3
export CHECKPOINT_BASE=${CHECKPOINT_BASE:-/checkpoint/$USER/fairdiplomacy}/$NAME
# shift off the first 3 arguments; any additional args get passed to the script
shift 3

mkdir -p $CHECKPOINT_BASE

for POWER in AUSTRIA ENGLAND FRANCE GERMANY ITALY RUSSIA TURKEY; do
    for SEED in $(seq $NUM_TRIALS); do
		POWERSEED="$(echo $POWER | head -c3).${SEED}"
        NAME="${NAME}_${POWERSEED}" $LAUNCH $AGENT_ONE $AGENT_SIX $POWER --seed $SEED --out ${CHECKPOINT_BASE}/game_${POWERSEED}.json $@
    done
done
