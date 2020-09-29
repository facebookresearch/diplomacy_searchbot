#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "resume_message_history_shortstate_allorder_chunk_batchsize4_priority"
NUM_HOURS = 72

# Define param grid
grid = {
    "--datapath": ["/private/home/wyshi/ParlAI/data"],
    "--load-from-checkpoint": ["True"],
    "--dynamic-batching": ["full"],
    "-t": ["message_history_shortstate_allorder_chunk"],
    "-dt": ["train:stream"],
    "--num-epochs": [20],
    "-veps": [0.1],
    "--attention-dropout": [0.00],
    "--dropout": [0.1],
    "--fp16": [True],
    "-m": ["bart",],
    "--n-positions": [1024],
    "--init-model": [
        "/checkpoint/wyshi/diplomacy/message_history_shortstate_allorder_chunk/202009211412_copy/model",
    ],
    "--dict_file": [
        "/checkpoint/wyshi/diplomacy/message_history_shortstate_allorder_chunk/202009211412_copy/model.dict"
    ],
    "--label-truncate": [512],
    "--log_every_n_secs": [10],
    "-lr": [2.37e-5],
    "--lr-scheduler": ["linear"],
    "--max-lr-steps": [150_000],
    "--lr-scheduler-patience": [3],
    "--optimizer": ["adam"],
    "--relu-dropout": [0.0],
    "--activation": ["gelu"],
    "--model-parallel": [False],
    "--save-after-valid": [True],
    "--text-truncate": [1024],
    "--warmup-updates": [0],
    "--fp16-impl": ["mem_efficient"],
    "--update-freq": [2],
    "--gradient-clip": [0.1],
    "--skip-generation": [True],
    "-vp": [100],
    "-vmt": ["ppl"],
    "-vmm": ["min"],
    "-stim": [360],
    "-vme": [10_000],
    "-bs": [4],
}

if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=sweep_name,
        partition="priority",
        comment="end of internship, last day 09/25",
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
