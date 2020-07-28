#!/bin/bash
#
# Example:
# NUM_TRIALS=2 slurm/compare_agents.sh mila mila compare_test1

set -e
set -u

ROOT="$(dirname $0)/.."

GPU=${GPU:-1}
CPU=${CPU:-10}
NUM_TRIALS=${NUM_TRIALS:-16}
NAME=${NAME:-dummy}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/checkpoint/$USER/fairdiplomacy/situation_check}/$NAME
mkdir -p $CHECKPOINT_DIR
PARTITION=${PARTITION:-learnfair}
MEM=${MEM:-0}
TIME=${TIME:-2880}

SBATCH_ARGS="--partition=$PARTITION --cpus-per-task=$CPU \
             --gpus=$GPU --mem=$MEM --time=$TIME --open-mode=append \
             --array=0-$(($NUM_TRIALS - 1))"



# log some stuff locally to make it easier to see what's going on
echo $(env) > $CHECKPOINT_DIR/env.inp
git log --graph --decorate --pretty=oneline --abbrev-commit --all > $CHECKPOINT_DIR/gitlog.inp
git diff $ROOT > $CHECKPOINT_DIR/gitdiff.inp

for REP in {0..1}; do
  FULL_NAME="${NAME}.${REP}"
  
  cat <<EOF | sbatch --job-name $FULL_NAME $SBATCH_ARGS
#!/bin/bash

python run.py \
  --adhoc \
  --cfg=$(pwd)/conf/c06_situation_check/cmp.prototxt \
  situation_json=$(pwd)/test_situations.json \
  seed=\$SLURM_ARRAY_TASK_ID \
  I.agent=agents/cfr1p_webdip \
  agent.cfr1p.model_path=/checkpoint/alerer/fairdiplomacy/sl_fbdata_20200717_minscore0/checkpoint.pth.best \
  agent.cfr1p.use_final_iter=false \
  agent.cfr1p.reset_seed_on_rollout=true \
  $@ \
  > ${CHECKPOINT_DIR}/game_\$SLURM_ARRAY_TASK_ID.$REP.log 2>&1
EOF
done
