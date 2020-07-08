#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()

# Params
sweep_name = 'diplomacy_basic_baseline_400M'

# Define param grid
grid = {
    '-t': ['dialogue'],
    '--min-turns': [3,],
    '-veps': [0.1],
    '--attention-dropout': [0.00],
    '--dropout': [0.1],
    '--fp16': [True],
    '--init-model': [
        '/checkpoint/edinan/diplomacy/400M_0701/model --dict-file /checkpoint/edinan/diplomacy/400M_0701/model.dict '
        '-m transformer/generator --embedding-size 1024 --ffn-size 4096 --attention-dropout 0.1 '
        '--n-heads 16 --n-positions 2048 --variant prelayernorm --activation gelu --n-encoder-layers 2 '
        '--n-decoder-layers 22 --skip-generation True --fp16 True --fp16-impl mem_efficient --force-fp16-tokens True '
        '--optimizer mem_eff_adam --truncate 128 --dict-tokenizer bytelevelbpe '
        '--bpe-vocab /checkpoint/edinan/20200625/reddit-400M-baseline/de8/model.dict-vocab.json '
        '--bpe-merge /checkpoint/edinan/20200625/reddit-400M-baseline/de8/model.dict-merges.txt'
    ],
    '--label-truncate': [128],
    '--log_every_n_secs': [10],
    '-lr': [7e-6, 1e-5],
    '--lr-scheduler': ['reduceonplateau'],
    '--lr-scheduler-patience': [3],
    '--optimizer': ['adam'],
    '--relu-dropout': [0.0],
    '--activation': ['gelu'],
    '--model-parallel': [True],
    '--save-after-valid': [True],
    '--text-truncate': [128],
    '--truncate': [128],
    '--warmup_updates': [100],
    '--fp16-impl': ['mem_efficient'],
    '--update-freq': [2],
    '--gradient-clip': [0.1],
    '--skip-generation': [True],
    '-vp': [10],
    '--max-train-time': [0.96 * 8 * 60 * 60],  # just under 8 hours
    '-vmt': ['ppl'],
    '-vmm': ['min'],
    '-stim': [360],
    '-vme': [10000],
    '-bs': [128],
}

if __name__ == '__main__':
    run_grid(
        grid=grid,
        name_keys={},
        sweep_name=sweep_name,
        partition='learnfair',
        jobtime='24:00:00',
        gpus=8,
        nodes=1,
        create_model_file=True,
        data_parallel=True,
        requeue=True,
        include_job_id=False,
        volta32=True,
        hashname=True,
        mem_gb=400,
        copy_env=False,
    )
