#!/bin/bash
#
# Example:
# NUM_TRIALS=2 slurm/compare_agents.sh mila mila compare_test1

set -e
set -u

ROOT="$(dirname $0)/.."
ROOT=$(realpath $ROOT)

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 agent_one agent_six name [script_args]"
  exit 1
fi

AGENT_ONE=$1
AGENT_SIX=$2
NAME=$3

GPU=${GPU:-1}
CPU=${CPU:-10}
NUM_TRIALS=${NUM_TRIALS:-10}
ARRAY_BASE=${ARRAY_BASE:-0}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/checkpoint/$USER/fairdiplomacy}/$NAME
PARTITION=${PARTITION:-learnfair}
MEM=${MEM:-0}
TIME=${TIME:-2880}
ARRAY_FILL_MISSING=${ARRAY_FILL_MISSING:-0}

# shift off the first 3 arguments; any additional args get passed to the script
shift 3

SBATCH_ARGS="--partition=$PARTITION --cpus-per-task=$CPU \
             --gpus=$GPU --mem=$MEM --time=$TIME --open-mode=append \
             --chdir=$CHECKPOINT_DIR"

ARRAY_MAX="$(($ARRAY_BASE + $NUM_TRIALS - 1))"
SBATCH_ARRAY_ARG="--array=$ARRAY_BASE-$ARRAY_MAX"
SRUN_MODE_ARG=""

# copy repo
CODE_CHECKPOINT=${CODE_CHECKPOINT:-"$($(dirname $0)/checkpoint_repo.sh)"}

# log some stuff locally to make it easier to see what's going on
mkdir -p $CHECKPOINT_DIR
[ -e $CHECKPOINT_DIR/code ] || ln -sf $CODE_CHECKPOINT $CHECKPOINT_DIR/code
echo $(env) > $CHECKPOINT_DIR/env.inp
git log --graph --decorate --pretty=oneline --abbrev-commit --all > $CHECKPOINT_DIR/gitlog.inp
git diff $ROOT > $CHECKPOINT_DIR/gitdiff.inp

cd $CHECKPOINT_DIR


if [ ! -e "conf" ]; then
  # HeyHi expects to see conf/ in the root folder.
  ln -s code/conf
fi


for POWER in AUSTRIA ENGLAND FRANCE GERMANY ITALY RUSSIA TURKEY; do
    POW="$(echo $POWER | head -c3)"
    FULL_NAME="${NAME}_${POW}"

    if [ $ARRAY_FILL_MISSING -ne 0 ]
    then
        MISSING_IDS=$($ROOT/bin/find_missing_game_jsons.py $CHECKPOINT_DIR $ARRAY_BASE $ARRAY_MAX $POWER)
        [ -z $MISSING_IDS ] && continue
        SBATCH_ARRAY_ARG="--array=$MISSING_IDS"
        SRUN_MODE_ARG="--mode start_continue"
    fi

    cat <<EOF | sbatch --job-name $FULL_NAME $SBATCH_ARGS $SBATCH_ARRAY_ARG
#!/bin/bash

OUT_DIR=$CHECKPOINT_DIR/${FULL_NAME}.\$SLURM_ARRAY_TASK_ID
mkdir -p \$OUT_DIR

PYTHONPATH=$CODE_CHECKPOINT srun --output \$OUT_DIR/stdout.log --error \$OUT_DIR/stderr.log -- \
python $CODE_CHECKPOINT/run.py $SRUN_MODE_ARG \
  --cfg=$CODE_CHECKPOINT/conf/c01_ag_cmp/cmp.prototxt \
  --exp_id_pattern_override=\$OUT_DIR \
  I.agent_one=agents/$AGENT_ONE \
  I.agent_six=agents/$AGENT_SIX \
  power_one=$POWER \
  seed=\$(echo "(\$SLURM_ARRAY_TASK_ID + 1000 * \$SLURM_ARRAY_JOB_ID) % 67867979" | bc) \
  out=${CHECKPOINT_DIR}/game_${POW}.\$SLURM_ARRAY_TASK_ID.json $@ \
  use_default_requeue=true

EOF
done
