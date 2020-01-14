#!/bin/bash
TMPDIR=$(mktemp -d -p /checkpoint/jsgray/tmp)

GIT_ROOT=../../

rsync -avz \
    --exclude-from $GIT_ROOT/.gitignore \
    --exclude .git \
    --exclude '*.db' \
    --exclude fairdiplomacy/data/out/ \
    --exclude fairdiplomacy/data/mila_dataset/ \
    $GIT_ROOT $TMPDIR

cd $TMPDIR/fairdiplomacy/compare_agents

for POWER in AUSTRIA ENGLAND FRANCE GERMANY ITALY RUSSIA TURKEY 
do

    for SEED in $(seq 2)
    do
        ##############
        ###   1v6  ###
        ##############

        cp sbatch.sh.pre sbatch.1v6.$POWER.$SEED.sh

        cat <<END >>sbatch.1v6.$POWER.$SEED.sh
srun --label \
    python compare_agents.py search mila $POWER \
    --out $TMPDIR/game.1v6.$POWER.$SEED.json \
    1>\$CHECKPOINT_DIR/stdout.log \
    2>\$CHECKPOINT_DIR/stderr.log
END

        PYTHONPATH=$TMPDIR sbatch sbatch.1v6.$POWER.$SEED.sh


        ##############
        ###   6v1  ###
        ##############

        cp sbatch.sh.pre sbatch.6v1.$POWER.$SEED.sh

        cat <<END >>sbatch.6v1.$POWER.$SEED.sh
srun --label \
    python compare_agents.py mila search $POWER \
    --out $TMPDIR/game.6v1.$POWER.$SEED.json \
    1>\$CHECKPOINT_DIR/stdout.log \
    2>\$CHECKPOINT_DIR/stderr.log
END

        PYTHONPATH=$TMPDIR sbatch sbatch.6v1.$POWER.$SEED.sh
    done
done

echo $TMPDIR
