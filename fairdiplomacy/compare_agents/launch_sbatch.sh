#!/bin/bash


#################
###   SETUP   ###
#################

AGENT_A=/checkpoint/jsgray/dipnet.pth
AGENT_B=search
NUM_TRIALS=10


################
###   IMPL   ###
################

TMPDIR=$(mktemp -d -p /checkpoint/$USER/tmp)

GIT_ROOT=../../

rsync -avz \
    --exclude-from $GIT_ROOT/.gitignore \
    --exclude .git \
    --exclude '*.db' \
    --exclude fairdiplomacy/data/out/ \
    --exclude fairdiplomacy/data/mila_dataset/ \
    --exclude thirdparty/github/fairinternal/torchbeast/ \
    $GIT_ROOT $TMPDIR

cd $TMPDIR/fairdiplomacy/compare_agents

echo $AGENT_A > AGENT_A.arg
echo $AGENT_B > AGENT_B.arg

for POWER in AUSTRIA ENGLAND FRANCE GERMANY ITALY RUSSIA TURKEY 
do

    for SEED in $(seq $NUM_TRIALS)
    do
        ###############
        ###   1v6   ###
        ###############

        cp sbatch.sh.pre sbatch.1v6.$POWER.$SEED.sh

        cat <<END >>sbatch.1v6.$POWER.$SEED.sh
echo $TMPDIR > \$CHECKPOINT_DIR/tmpdir.path
srun --label \
    python compare_agents.py $AGENT_A $AGENT_B $POWER \
    --seed $SEED \
    --out game.1v6.$POWER.$SEED.json \
    1>\$CHECKPOINT_DIR/stdout.log \
    2>\$CHECKPOINT_DIR/stderr.log
END

        PYTHONPATH=$TMPDIR WORKING_DIR=/private/home/jsgray/.mila_working_dir \
            sbatch sbatch.1v6.$POWER.$SEED.sh


        ###############
        ###   6v1   ###
        ###############

        cp sbatch.sh.pre sbatch.6v1.$POWER.$SEED.sh

        cat <<END >>sbatch.6v1.$POWER.$SEED.sh
echo $TMPDIR > \$CHECKPOINT_DIR/tmpdir.path
srun --label \
    python compare_agents.py $AGENT_B $AGENT_A $POWER \
    --seed $SEED \
    --out game.6v1.$POWER.$SEED.json \
    1>\$CHECKPOINT_DIR/stdout.log \
    2>\$CHECKPOINT_DIR/stderr.log
END

        PYTHONPATH=$TMPDIR WORKING_DIR=/private/home/jsgray/.mila_working_dir \
            sbatch sbatch.6v1.$POWER.$SEED.sh

    done
done

echo $TMPDIR
