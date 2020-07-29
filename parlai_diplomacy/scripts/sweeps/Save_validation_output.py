#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "diplomacy_validation_save_split"
NUM_HOURS = 48

# trick for test file id
TEST_SPLIT = 64
test_file_ids = []
REPORT_FILE_PREFIX = (
    f"/checkpoint/fairdiplomacy/validation_report/{diplomacy_validation_save_split}_output_split"
)
test_file_ids = [
    f"{i} --report-filename {REPORT_FILE_PREFIX}_{i}" for i in range(1, TEST_SPLIT + 1)
]


# Define param grid
grid = {
    "-mf": [
        "/checkpoint/wyshi/20200709/diplomacy_All_in_one_context512_batch128_truncate1024_baseline/90a/model"
    ],
    "-t": ["message_order"],
    "--save-world-logs": [True],
    "-dt": ["valid"],
    "--skip-generation": [False],
    "--inference": ["greedy"],
    "--include-message-from": ["partner_msg_only"],
    "--input-seq-content": ["state_only"],
    "-bs": [256],
    "--min-turns": [1,],
    "--split-valid": ["yes"],
    "--test-file-id": test_file_ids,
    "--dynamic-batching": ["full"],
}

if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=sweep_name,
        partition="learnfair",
        jobtime=f"{NUM_HOURS}:00:00",
        prefix="python -u examples/eval_model.py",
        gpus=1,
        nodes=1,
        create_model_file=False,
        data_parallel=True,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        copy_env=True,
    )
