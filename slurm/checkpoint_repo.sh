#!/bin/bash
set -e

mkdir -p /checkpoint/$USER/tmp
TMPDIR=$(mktemp -d -p /checkpoint/$USER/tmp)
ROOT="$(realpath $(dirname $0)/..)"
TARBALL=repo.tar

echo "Syncing $ROOT --> $TMPDIR" 1>&2
tar -C $ROOT -cf $TMPDIR/$TARBALL \
    --exclude-vcs \
    --exclude .git \
    --exclude '*.db' \
    --exclude fairdiplomacy/data/out/ \
    --exclude thirdparty/github/fairinternal/postman \
    --exclude thirdparty/github/diplomacy/diplomacy/diplomacy/web \
    --exclude .mypy_cache \
    .
tar -C $TMPDIR -xf $TMPDIR/$TARBALL
rm $TMPDIR/$TARBALL

echo $TMPDIR
