# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Creates or activates an environment of fair or AWS cluster.
# Usage:
#  . fair_activate.sh
if hostname | grep -q fair; then
    # Fair cluster, not needed/supported for AWS.
    module load anaconda3/5.0.1
    #module load cudnn/v7.3-cuda.9.2
    module load cudnn/v7.6.5.32-cuda.10.1
    #module load cuda/9.2
    module load cuda/10.1
    module load singularity/3.4.1/gcc.7.3.0
fi

# The name of the environment is the name of the project folder. Could be
# redefined if one wants to use several repos with the same conda.
xxx_env_name="${FORCE_ENV_NAME:-"$(basename "$(pwd)")"}"

if ! conda env list --json | jq ".envs | .[]" | grep -qE "/${xxx_env_name}\""; then
    conda create --yes -n $xxx_env_name python=3.7
    source activate $xxx_env_name
    git submodule init
    git submodule update
    make deps all || {
        echo
        echo "################################################"
        echo "## Failed to install deps                      #"
        echo "## To re-run install: make deps all            #"
        echo "################################################"
    }
fi
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
export OMP_NUM_THREADS=1
source activate $xxx_env_name
ulimit -c unlimited
