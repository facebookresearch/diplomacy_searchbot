#!/bin/bash
mkdir -p /checkpoint/$USER/tmp
TMPDIR=$(mktemp -d -p /checkpoint/$USER/tmp)

rsync -avz \
    --exclude-from ../../../.gitignore \
    --exclude .git \
    --exclude '*.db' \
    --exclude fairdiplomacy/data/out/ \
    --exclude fairdiplomacy/data/mila_dataset/ \
    ../../../ $TMPDIR

cd $TMPDIR/fairdiplomacy/models/dipnet
PYTHONPATH=$TMPDIR sbatch train_sl_sbatch.sh
echo $TMPDIR
