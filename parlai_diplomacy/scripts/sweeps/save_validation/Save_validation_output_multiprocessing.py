#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "validation_diplomacy_multiprocess_one_job_per_node=False_bs128"
NUM_HOURS = 72

# Define param grid
grid = {
    "--datapath": ["/private/home/wyshi/ParlAI/data"],
    "-mf": ["/checkpoint/wyshi/20200730/Bart_diplomacy_lower_lr/18e/model"],
    "-m": ["bart",],
    "-t": ["state_order_chunk"],
    "-dt": ["valid:stream"],
    "--report-filename": [
        "/checkpoint/fairdiplomacy/press_diplomacy/validation/validation_report/test_multiprocess_128"
    ],
    "--label-truncate": [256],
    "--text-truncate": [1024],
    "--save-world-logs": [True],
    "--skip-generation": [False],
    "--inference": ["greedy"],
    "-bs": [128],
    "--min-turns": [1,],
}

if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=sweep_name,
        partition="learnfair",
        jobtime=f"{NUM_HOURS}:00:00",
        # prefix="python -u parlai_diplomacy/scripts/distributed_eval.py",
        prefix="python -u parlai_diplomacy/scripts/multiprocessing_eval.py",
        one_job_per_node=False,
        gpus=8,
        # cpus=80,
        # nodes=8,
        nodes=1,
        create_model_file=False,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        copy_env=True,
    )
