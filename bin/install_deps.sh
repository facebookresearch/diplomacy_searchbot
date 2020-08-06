#!/bin/bash
set -e

ROOT=$(dirname $0)/..
cd $ROOT

set -x

git submodule update
conda install -y nodejs
conda install --yes -c esri pybind11
if [ -z "$CIRCLECI" ]; then
    conda install --yes pytorch=1.4 torchvision cudatoolkit=9.2 -c pytorch
else
    echo "Skipping pytorch build on CIRCLECI"
fi
if hostname | grep -q devfair; then
    # Install pytorch and patch cmake files so that things, e.g., postman, could be built against torch.
    sed -i -e 's#/usr/local/cuda/lib64/libnvToolsExt.so#/public/apps/cuda/9.2/lib64/libnvToolsExt.so#g' $CONDA_PREFIX/lib/python3.7/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
    sed -i -e 's#/usr/local/cuda/lib64/libcudart.so#/public/apps/cuda/9.2/lib64/libcudart.so#g' $CONDA_PREFIX/lib/python3.7/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
    sed -i -e 's#/usr/local/cuda/lib64/libculibos.a#/public/apps/cuda/9.2/lib64/libculibos.a#g' $CONDA_PREFIX/lib/python3.7/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
    pip install -e "git+ssh://git@github.com/fairinternal/submitit@master#egg=submitit"
else
    pip install submitit
fi
pip install -e ./thirdparty/github/fairinternal/postman/nest/
pip install -e ./thirdparty/github/fairinternal/postman/postman/
pip install -e ./thirdparty/github/diplomacy/diplomacy
pip install -e ./dipcc
pip install -e . -vv
