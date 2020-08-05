#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
import parlai_diplomacy.utils.loading as load
import parlai_diplomacy.tasks.language_diplomacy.utils as utls

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = "diplomacy_AllInOne_SpecialToken_baseline"
NUM_HOURS = 72

special_tokens, _ = utls.load_special_tokens()
special_tokens_text = ",".join(special_tokens)

# Define param grid
grid = {
    "--hf-skip-special-tokens": [False],
    "--special-tok-lst": [special_tokens_text],
    "--with-special-token": [True],
    "--load-from-checkpoint": ["True"],
    "--dynamic-batching": ["full"],
    "-t": ["message_state_order"],
    "--num-epochs": [5],
    "--min-turns": [1,],
    "--include-message-from": ["all_msg",],
    "-veps": [0.1],
    "--attention-dropout": [0.00],
    "--dropout": [0.1],
    "--fp16": [True],
    "--init-model": [
        "/checkpoint/wyshi/diplomacy/context512_test_model/model --dict-file /checkpoint/wyshi/diplomacy/context512_test_model/model.dict "
        "-m transformer/generator --embedding-size 1024 --ffn-size 4096 --attention-dropout 0.1 "
        "--n-heads 16 --n-positions 2048 --variant prelayernorm --activation gelu --n-encoder-layers 2 "
        "--n-decoder-layers 22 --skip-generation True --fp16 True --fp16-impl mem_efficient --force-fp16-tokens True "
        "--optimizer mem_eff_adam --dict-tokenizer bytelevelbpe "
        "--bpe-vocab /checkpoint/wyshi/diplomacy/context512_test_model/model.dict-vocab.json "
        "--bpe-merge /checkpoint/wyshi/diplomacy/context512_test_model/model.dict-merges.txt"
    ],
    "--label-truncate": [256],
    "--log_every_n_secs": [10],
    "-lr": [1e-5, 5e-5],
    "--lr-scheduler": ["reduceonplateau"],  # linear
    "--lr-scheduler-patience": [3],
    "--optimizer": ["adam"],
    "--relu-dropout": [0.0],
    "--activation": ["gelu"],
    "--model-parallel": [False],
    "--save-after-valid": [True],
    "--text-truncate": [2048],
    "--truncate": [2048],
    "--warmup-updates": [5000],  # larger, 1000
    "--fp16-impl": ["mem_efficient"],
    "--update-freq": [2],
    "--gradient-clip": [0.1],
    "--skip-generation": [True],
    "-vp": [10],
    "--max-train-time": [0.96 * NUM_HOURS * 60 * 60],  # just under 8 hours
    "-vmt": ["ppl"],
    "-vmm": ["min"],
    "-stim": [360],
    "-vme": [10000],
    "-bs": [32],
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
