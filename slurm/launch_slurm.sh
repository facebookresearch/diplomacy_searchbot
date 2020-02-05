#!/bin/bash

# This is a helper script that launches a job on slurm.
# it is primarily meant to be called from another script that runs a sweep,
# NOT run directly from the command line. See slurm/example_train_sl.sh
# for more details.

set -e


ROOT="$(dirname $0)/.."

SCRIPT=${SCRIPT:-fairdiplomacy/models/dipnet/train_sl.py}
PARTITION=${PARTITION:-learnfair}
GPU=${GPU:-8}
CPU=${CPU:-80}
MEM=${MEM:-0}
TIME=${TIME:-2880}

if [ x"$NAME" == x ]; then
  echo "Must specify NAME env var."
  exit 1
fi

CHECKPOINT_DIR="/checkpoint/$USER/fairdiplomacy/$NAME"
if [ -e $CHECKPOINT_DIR ]; then
  echo "$CHECKPOINT_DIR already exists."
  exit 1
fi

SRUN_DEFAULTS="--job-name $NAME --partition=$PARTITION --cpus-per-task=$CPU \
               --gres=gpu:$GPU --mem=$MEM --time=$TIME --open-mode=append \
               --chdir=$CHECKPOINT_DIR \
               --output $CHECKPOINT_DIR/stdout.log --error $CHECKPOINT_DIR/stderr.log"

if [ x"$CODE_CHECKPOINT" == x ]; then
  CODE_CHECKPOINT=$($ROOT/slurm/checkpoint_repo.sh)
fi


mkdir -p $CHECKPOINT_DIR

# log some stuff locally to make it easier to see what's going on
ln -sf $CODE_CHECKPOINT $CHECKPOINT_DIR/code
echo "$SCRIPT $@" > $CHECKPOINT_DIR/args.inp
echo $(env) > $CHECKPOINT_DIR/env.inp
git log --graph --decorate --pretty=oneline --abbrev-commit --all > $CHECKPOINT_DIR/gitlog.inp
git diff $ROOT > $CHECKPOINT_DIR/gitdiff.inp

cd $CHECKPOINT_DIR

PYTHONPATH=$CODE_CHECKPOINT srun $SRUN_DEFAULTS $SRUN_ARGS -- python $CODE_CHECKPOINT/$SCRIPT $@ &

echo "Launched $NAME"
