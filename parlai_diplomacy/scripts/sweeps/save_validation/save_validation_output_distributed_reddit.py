#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "debug_meena_save_dynamic_batching"
NUM_HOURS = 72

# Define param grid
grid = {
    "--verbose": [True],
    "--datapath": ["/private/home/wyshi/ParlAI/data"],
    "-mf": ["/checkpoint/wyshi/diplomacy/Bart/Bart_diplomacy_lower_lr/18e/model"],
    "-m": ["bart",],
    "-t": ["meena_data_v2"],
    "-dt": ["valid:stream"],
    # "--report-filename": [
    #     "/checkpoint/fairdiplomacy/press_diplomacy/validation/validation_report/debug/test_distributed_meena"
    # ],
    "--label-truncate": [256],
    "--text-truncate": [1024],
    # "--save-world-logs": [True],
    "--skip-generation": [True],
    "--dynamic-batching": ["full"],
    "--inference": ["greedy"],
    "-bs": [128],
}

if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=sweep_name,
        partition="learnfair",
        jobtime=f"{NUM_HOURS}:00:00",
        prefix="python -u parlai_diplomacy/scripts/distributed_eval.py",
        # prefix="python -u parlai_diplomacy/scripts/multiprocessing_eval.py",
        one_job_per_node=False,
        gpus=8,
        nodes=8,
        # nodes=1,
        create_model_file=False,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        copy_env=True,
    )
