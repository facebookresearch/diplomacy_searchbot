#!/bin/bash
set -e


ROOT=$(dirname $0)/..
cd $ROOT

protoc conf/conf.proto --python_out .

git submodule update
pip install -e ./thirdparty/github/fairinternal/postman/postman/
