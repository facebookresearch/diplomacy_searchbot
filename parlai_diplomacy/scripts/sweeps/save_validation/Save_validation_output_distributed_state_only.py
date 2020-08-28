#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid, bash
import parlai_diplomacy.utils.loading as load
import parlai_diplomacy.scripts.evaluation.aggregate_valid_reports as aggregate_valid_reports
from parlai.utils import logging

from datetime import date
import os

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "eval_shortstate_order_chunk_BART_olddata_oldcode_parlai_prev0.8"  # change this
NUM_HOURS = 72

# save path
DATE = str(date.today()).replace("-", "")
TEACHER = "shortstate_order_chunk"  # change this
(
    VALID_REPORT_SAVE_PATH,
    _,
    VALID_REPORT_SAVE_JSON_PATH,
) = aggregate_valid_reports.get_valid_report_path(sweep_name, TEACHER, date_of_sweep=DATE)
logging.info(f"mkdir {VALID_REPORT_SAVE_PATH}")
bash("mkdir -p " + VALID_REPORT_SAVE_PATH)
# predict_all_order = "--predict_all_order"

# Define param grid
grid = {
    # "--verbose": [True],
    "--datapath": ["/private/home/wyshi/ParlAI/data"],
    "-mf": [
        "/checkpoint/wyshi/20200825/shortstate_order_chunk_Bart_ParlaiPre0.8.0_33d07c1/de8/model"
    ],  # change this
    "-m": ["bart",],
    "-t": [f"{TEACHER}"],
    "-dt": ["valid:stream"],
    "--report-filename": [os.path.join(VALID_REPORT_SAVE_PATH, "valid_report")],
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
        partition="dev",
        jobtime=f"{NUM_HOURS}:00:00",
        prefix="python -u parlai_diplomacy/scripts/distributed_eval.py",
        one_job_per_node=False,
        gpus=8,
        nodes=2,
        create_model_file=False,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        copy_env=True,
    )

    # bash(
    #     f"python scripts/evaluation/aggregate_valid_reports.py --sweep_name {sweep_name} --teacher {TEACHER} --date_of_sweep {DATE} {predict_all_order}"
    # )
    # bash(
    #     f"python scripts/evaluation/compute_accuracy.py --eval_file {VALID_REPORT_SAVE_JSON_PATH} --json"
    # )
