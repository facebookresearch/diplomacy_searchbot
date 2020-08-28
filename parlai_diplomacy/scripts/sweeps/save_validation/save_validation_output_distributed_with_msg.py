#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
import parlai_diplomacy.utils.loading as load
import parlai_diplomacy.scripts.aggregate_valid_reports as aggregate_valid_reports

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "save_diplomacy_validation_bart_with_msg"
NUM_HOURS = 72

# save path
TEACHER = "message_history_state_order_chunk"
DATA_TYPE = "valid:stream"
REPORT_SAVE_PATH, _, _ = aggregate_valid_reports.get_valid_report_path(
    sweep_name, DATA_TYPE, TEACHER
)
bash("mkdir -p " + REPORT_SAVE_PATH)

# Define param grid
grid = {
    "--verbose": [True],
    "--datapath": ["/private/home/wyshi/ParlAI/data"],
    "-mf": ["/checkpoint/wyshi/diplomacy/Bart/Bart_diplomacy/37a/model"],
    "-m": ["bart",],
    "-t": ["message_history_state_order_chunk"],
    "-dt": [DATA_TYPE],
    "--report-filename": [REPORT_SAVE_PATH],
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
        one_job_per_node=False,
        gpus=8,
        nodes=8,
        create_model_file=False,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        copy_env=True,
    )
