#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid, bash
import parlai_diplomacy.utils.loading as load
import parlai_diplomacy.tasks.common_task_utils as utls
import parlai_diplomacy.utils.valid_result_converting as valid_result_converting

from datetime import date

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "save_diplomacy_validation_bart"
NUM_HOURS = 72

# save path
TEACHER = "state_order_chunk"
VALID_REPORT_SAVE_PATH, _, _ = valid_result_converting.get_valid_report_path(sweep_name, TEACHER)
bash("mkdir -p " + VALID_REPORT_SAVE_PATH)

# Define param grid
grid = {
    "--verbose": [True],
    "--datapath": ["/private/home/wyshi/ParlAI/data"],
    "-mf": ["/checkpoint/wyshi/diplomacy/Bart/Bart_diplomacy_lower_lr/18e/model"],
    "-m": ["bart",],
    "-t": [f"{TEACHER}"],
    "-dt": ["valid:stream"],
    "--report-filename": [VALID_REPORT_SAVE_PATH],
    "--label-truncate": [256],
    "--text-truncate": [1024],
    "--save-world-logs": [True],
    "--skip-generation": [False],
    # "--dynamic-batching": ["full"],
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
