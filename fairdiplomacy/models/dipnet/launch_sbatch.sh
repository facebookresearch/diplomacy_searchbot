#!/bin/bash
TMPDIR=$(mktemp -d -p /checkpoint/jsgray/tmp)

rsync -avz \
    --exclude-from ../../../.gitignore \
    --exclude .git \
    --exclude '*.db' \
    --exclude fairdiplomacy/data/out/ \
    --exclude fairdiplomacy/data/mila_dataset/ \
    ../../../ $TMPDIR

sbatch $TMPDIR/fairdiplomacy/models/dipnet/train_sl_sbatch.sh
echo $TMPDIR
