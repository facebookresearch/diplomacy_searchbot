#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "BART_diplomacy_dialogue"
NUM_HOURS = 72

# Define param grid
grid = {
    "--datapath": ["/private/home/apjacob/ParlAI/data"],
    "--load-from-checkpoint": ["True"],
    "--dynamic-batching": ["full"],
    "-t": [
        "state_message_history_dialogue_chunk",
        "message_history_state_dialogue_chunk",
        "message_history_dialogue_chunk",
    ],
    "-dt": ["train:stream"],
    "--num-epochs": [10],
    "--min-turns": [1,],
    # "--include-message-from": ["all_msg",],
    "-veps": [0.1],
    "--attention-dropout": [0.00],
    "--dropout": [0.1],
    "--fp16": [True],
    "-m": ["bart",],
    "--init-model": ["/private/home/apjacob/ParlAI/data/models/bart/bart_large/model",],
    "--n-positions": [1024],
    "--dict_file": ["/private/home/apjacob/ParlAI/data/models/bart/bart_large/model.dict"],
    "--label-truncate": [320],
    "--log_every_n_secs": [10],
    "-lr": [1e-5, 5e-5, 1e-4],
    "--lr-scheduler": ["linear"],
    "--max-lr-steps": ["50_000"],
    "--lr-scheduler-patience": [3],
    "--optimizer": ["adam"],
    "--relu-dropout": [0.0],
    "--activation": ["gelu"],
    "--model-parallel": [False],
    "--save-after-valid": [True],
    "--text-truncate": [800],
    "--warmup-updates": [5000],
    "--fp16-impl": ["mem_efficient"],
    "--update-freq": [1],
    "--gradient-clip": [0.1],
    "--skip-generation": [True],
    "-vp": [50],
    # "--max-train-time": [0.96 * NUM_HOURS * 60 * 60],  # just under 8 hours
    "-vmt": ["ppl"],
    "-vmm": ["min"],
    "-stim": [360],
    "-vme": [10_000],
    "-bs": [8],
}

if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=sweep_name,
        partition="learnfair",
        jobtime=f"{NUM_HOURS}:00:00",
        prefix="python -u parlai_diplomacy/scripts/distributed_train.py",
        one_job_per_node=False,
        gpus=8,
        nodes=8,
        create_model_file=True,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        copy_env=True,
    )
