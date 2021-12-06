#!/bin/bash
set -e

TMPDIR=$(mktemp -d)
ROOT="$(realpath $(dirname $0)/..)"
TARBALL=repo.tar

echo "Syncing $ROOT --> $TMPDIR" 1>&2
tar -C $ROOT -cf $TMPDIR/$TARBALL \
    --exclude-vcs \
    --exclude .git \
    --exclude '*.db' \
    --exclude build/ \
    --exclude dipcc/ \
    --exclude integration_tests/ \
    --exclude unit_tests/ \
    --exclude src/ \
    --exclude fairdiplomacy/data/out/ \
    --exclude thirdparty/github/fairinternal/postman \
    --exclude thirdparty/github/diplomacy/diplomacy/diplomacy/web \
    --exclude .mypy_cache \
    --exclude local \
    .
tar -C $TMPDIR -xf $TMPDIR/$TARBALL
rm $TMPDIR/$TARBALL

(git rev-parse HEAD > $TMPDIR/GITHASH.txt) || true
(git diff HEAD > $TMPDIR/GITDIFF.txt) || true

echo $TMPDIR
