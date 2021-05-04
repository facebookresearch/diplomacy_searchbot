#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


MODE=${MODE:-Release}

pushd $(dirname $0)

cmake -DCMAKE_BUILD_TYPE=$MODE . && make -j ${N_DIPCC_JOBS:-}

popd >/dev/null
