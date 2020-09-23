#!/bin/bash -e

MODE=${MODE:-RELEASE}

pushd $(dirname $0)

cmake -DCMAKE_BUILD_TYPE=$MODE . && make -j ${N_DIPCC_JOBS:-}

popd >/dev/null
