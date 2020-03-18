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
    --exclude thirdparty/github/fairinternal/postman \
    --exclude thirdparty/github/diplomacy/diplomacy/diplomacy/web \
    $ROOT $TMPDIR

echo "$TMPDIR/fairdiplomacy"
