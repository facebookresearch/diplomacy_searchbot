#!/bin/bash

CODE_CHECKPOINT="$($(dirname $0)/checkpoint_repo.sh)"

OUT_DIR=/checkpoint/$USER/diplomacy/slurm/cfr_db_cache
mkdir -p $OUT_DIR

BASE=${BASE:-10000}
N=${N:-500}

cat <<EOF | sbatch --job-name cfr_db_cache \
                   --partition=learnfair \
                   --open-mode=append \
                   --chdir=$OUT_DIR \
                   --gpus=1 \
                   --cpus-per-task=40 \
                   --mem=200GB \
                   --constraint=pascal \
                   --time=2880 \
                   --array=0-$(($N - 1))
#!/bin/bash

PREFIX=\$(( $BASE + \$SLURM_ARRAY_TASK_ID ))
OUT_LOG=$OUT_DIR/\$PREFIX.log

PYTHONPATH=$CODE_CHECKPOINT srun --output \$OUT_LOG --error \$OUT_LOG -- \
python $CODE_CHECKPOINT/run.py --adhoc \
    --cfg=$CODE_CHECKPOINT/conf/c05_build_db_cache/build_db_cache.prototxt \
    I.cf_agent=agents/cfr1p \
    cf_agent.cfr1p.n_rollouts=100 \
    cf_agent.cfr1p.n_plausible_orders=24 \
    cf_agent.cfr1p.n_rollout_procs=168 \
    cf_agent.cfr1p.average_n_rollouts=3 \
    out_path=$OUT_DIR/cache_\$PREFIX.pt \
    only_with_min_final_score=0 \
    n_cf_agent_samples=10 \
    n_parallel_jobs=1 \
    glob=/checkpoint/jsgray/diplomacy/mila_dataset/data/game_\${PREFIX}?.json
EOF
