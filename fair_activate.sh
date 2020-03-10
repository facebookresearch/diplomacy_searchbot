# Creates or activates an environment of fair or AWS cluster.
# Usage:
#  . fair_activate.sh
if hostname | grep -q fair; then
    # Fair cluster, not needed/supported for AWS.
    module load anaconda3/5.0.1
    module load cudnn/v7.3-cuda.9.2
    module load cuda/9.2
    module load singularity/3.4.1/gcc.7.3.0
fi

# The name of the environment is the name of the project folder. Could be
# redefined if one wants to use several repos with the same conda.
xxx_env_name="${FORCE_ENV_NAME:-"$(basename "$(pwd)")"}"

if ! conda env list --json | jq ".envs | .[]" | grep -qE "/${xxx_env_name}\""; then
    conda create --yes -n $xxx_env_name python=3.7
    source activate $xxx_env_name
    conda install -y nodejs
    git submodule init
    git submodule update
    # Install pytorch and patch cmake files so that things, e.g., postman, could be built against torch.
    conda install --yes pytorch=1.4 torchvision cudatoolkit=9.2 -c pytorch
    sed -i -e 's#/usr/local/cuda/lib64/libnvToolsExt.so#/public/apps/cuda/9.2/lib64/libnvToolsExt.so#g' $CONDA_PREFIX/lib/python3.7/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
    sed -i -e 's#/usr/local/cuda/lib64/libcudart.so#/public/apps/cuda/9.2/lib64/libcudart.so#g' $CONDA_PREFIX/lib/python3.7/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
    sed -i -e 's#/usr/local/cuda/lib64/libculibos.a#/public/apps/cuda/9.2/lib64/libculibos.a#g' $CONDA_PREFIX/lib/python3.7/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
    pip install -e . -vv
    pip install -e ./thirdparty/github/fairinternal/postman/nest/
    pip install -e ./thirdparty/github/fairinternal/postman/src/postman/
    pip install -e "git+ssh://git@github.com/fairinternal/submitit@master#egg=submitit"
fi
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
export OMP_NUM_THREADS=1
source activate $xxx_env_name
ulimit -c unlimited
