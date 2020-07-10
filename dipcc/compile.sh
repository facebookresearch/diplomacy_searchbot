#!/bin/bash -e

MODE=${MODE:-RELEASE}

pushd $(dirname $0)

cmake -DCMAKE_BUILD_TYPE=$MODE -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=./dipcc/python . \
    && make -j

popd >/dev/null
