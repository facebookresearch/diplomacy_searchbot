#!/bin/bash
set -e


ROOT=$(dirname $0)/..
cd $ROOT

protoc conf/conf.proto --python_out .

make -C thirdparty/github/fairinternal/postman develop