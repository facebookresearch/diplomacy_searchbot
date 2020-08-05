#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid

HOURS = 2

# Params
sweep_name = "reddit_400M_longcontext_test_distributed"

# Define param grid
grid = {
    "-t": ["meena_data_v2"],
    "-dt": ["train:stream"],
    "-vtim": [900],
    "--attention-dropout": [0.1],
    "--batchsize": [8],
    "--eval-batchsize": [32],
    "--model": ["transformer/generator"],
    "--embedding-size": [1024],
    "--ffn-size": [4096],
    "--variant": ["prelayernorm"],
    "--n-heads": [16],
    "--n-positions": [2048],
    "--n-encoder-layers": [2],
    "--n-decoder-layers": [22],
    "--history-add-global-end-token": ["end"],
    "--dict-tokenizer": ["bytelevelbpe"],
    "--dict-file": ["zoo:blender/reddit_3B/model.dict"],
    "--dropout": [0.1],
    "--fp16": [True],
    "--label-truncate": [512],
    "--log_every_n_secs": [10],
    "-lr": [6e-4],
    "--lr-scheduler": ["invsqrt"],
    "--optimizer": ["adam"],
    "--relu-dropout": [0.0],
    "--activation": ["gelu"],
    "--model-parallel": ["false"],
    "--load-from-checkpoint": ["false"],
    "--text-truncate": [512, 1024],
    "--warmup_updates": [10000],
    "--fp16-impl": ["mem_efficient"],
    "--update-freq": [1],
    "--gradient-clip": [10.0],
    "--skip-generation": [True],
    "-vp": [10],
    "-vmt": ["ppl"],
    "-vmm": ["min"],
    "--dynamic-batching": ["full"],
    "-tblog": ["true"],
}

if __name__ == "__main__":
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=sweep_name,
        partition="learnfair",
        jobtime=f"{HOURS}:00:00",
        prefix="python -u parlai_diplomacy/scripts/distributed_train.py",
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
