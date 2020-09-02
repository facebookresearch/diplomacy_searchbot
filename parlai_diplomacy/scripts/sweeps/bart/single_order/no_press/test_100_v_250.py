#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "test1000v250"
NUM_HOURS = 72

# Define param grid
grid = {
    "--datapath": ["/private/home/edinan/ParlAI/data"],
    "--load-from-checkpoint": ["True"],
    "--dynamic-batching": ["full"],
    "-t": ["shortstate_order_chunk"],
    "--loading-chunks": [250, 1000],
    "-dt": ["train:stream"],
    "--num-epochs": [20],
    "-veps": [0.05],
    "--attention-dropout": [0.00],
    "--dropout": [0.1],
    "--fp16": [True],
    "-m": ["bart",],
    # "--init-model": [
    #     "/checkpoint/wyshi/20200826/newdata_shortstate_order_chunk_bart_diplomacy/18e/model.checkpoint",
    # ],
    "--n-positions": [1024],
    # "--dict_file": [
    #     "/checkpoint/wyshi/20200826/newdata_shortstate_order_chunk_bart_diplomacy/18e/model.dict"
    # ],
    "--label-truncate": [256],
    "--log_every_n_secs": [10],
    "-lr": [3.341e-05],
    "--lr-scheduler": ["linear"],
    "--max-lr-steps": [50_000 - 3341],
    "--lr-scheduler-patience": [3],
    "--optimizer": ["adam"],
    "--relu-dropout": [0.0],
    "--activation": ["gelu"],
    "--model-parallel": [False],
    "--save-after-valid": [True],
    "--text-truncate": [256],
    "--warmup-updates": [5_000 - 3341],
    "--fp16-impl": ["mem_efficient"],
    "--update-freq": [1],
    "--gradient-clip": [0.1],
    "--skip-generation": [True],
    "-vp": [100],
    "-vmt": ["ppl"],
    "-vmm": ["min"],
    "-stim": [360],
    "-vme": [10_000],
    "-bs": [32],
}

if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=sweep_name,
        partition="dev",
        jobtime=f"{NUM_HOURS}:00:00",
        prefix="python -u parlai_diplomacy/scripts/distributed_train.py",
        one_job_per_node=False,
        gpus=8,
        nodes=1,
        create_model_file=True,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        copy_env=True,
    )
