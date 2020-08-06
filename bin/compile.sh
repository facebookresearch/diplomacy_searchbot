#!/bin/bash
set -e


ROOT=$(dirname $0)/..
cd $ROOT

protoc conf/*.proto --python_out .
