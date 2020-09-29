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
sweep_name = "pseudo_labelling_correct_rerun"  # change this
NUM_HOURS = 72

# save path
DATE = str(date.today()).replace("-", "")
TEACHER = "pseudoorder_generation_message_history_shortstate_dialogue_chunk"  # change this
(
    VALID_REPORT_SAVE_PATH,
    _,
    VALID_REPORT_SAVE_JSON_PATH,
    _,
) = aggregate_valid_reports.get_valid_report_path(sweep_name, TEACHER, date_of_sweep=DATE)
logging.info(f"mkdir {VALID_REPORT_SAVE_PATH}")
bash("mkdir -p " + VALID_REPORT_SAVE_PATH)
# predict_all_order = "--predict_all_order"

# Define param grid
grid = {
    "--verbose": [True],
    # "--load_from_checkpoint": [False],
    "--datapath": ["/private/home/wyshi/ParlAI/data"],
    "-im": [
        "/checkpoint/wyshi/diplomacy/message_history_shortstate_allorder_chunk/202009200146/model.checkpoint"
    ],  # change this
    "-mf": [
        "/checkpoint/wyshi/diplomacy/message_history_shortstate_allorder_chunk/202009200146/model.checkpoint"
    ],  # change this
    "-m": ["bart",],
    "-t": [f"{TEACHER}"],
    "-dt": ["train:evalmode:stream"],
    "--report-filename": [os.path.join(VALID_REPORT_SAVE_PATH, "valid_report")],
    "--label-truncate": [512],
    "--text-truncate": [1024],
    "--save-world-logs": [True],
    "--skip-generation": [False],
    "--inference": ["greedy"],
    # "--dynamic-batching": ["full"],
    # "--n_chunks": [1],
    "--log-keep-fields": ["game_id,phase_id,player_id,eval_labels,example_id,player,text"],
    "-bs": [80],
}

if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=sweep_name,
        # partition="dev",
        partition="learnfair",
        # comment="end of internship, last day 09/25",
        jobtime=f"{NUM_HOURS}:00:00",
        prefix="python -u parlai_diplomacy/scripts/distributed_eval.py",
        one_job_per_node=False,
        gpus=8,
        nodes=32,
        create_model_file=False,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        copy_env=True,
    )
