#!/bin/bash
set -e

mkdir -p /checkpoint/$USER/tmp
TMPDIR=$(mktemp -d -p /checkpoint/$USER/tmp)
ROOT="$(realpath $(dirname $0)/..)"

echo "Syncing $ROOT --> $TMPDIR" 1>&2
rsync -az \
    --exclude-from $ROOT/.gitignore \
    --exclude .git \
    --exclude '*.db' \
    --exclude fairdiplomacy/data/out/ \
    --exclude fairdiplomacy/data/mila_dataset/ \
    --exclude thirdparty/github/fairinternal/postman \
    $ROOT $TMPDIR

echo $TMPDIR
