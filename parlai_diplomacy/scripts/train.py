#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a model using parlai's standard training loop.
For documentation, see parlai.scripts.train_model.

NOTE: we will have to import all relevant agents here
"""
from parlai.scripts.train_model import TrainModel
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()


if __name__ == "__main__":
    TrainModel.main()
