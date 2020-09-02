#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai_diplomacy.utils.param_sweeps.param_sweep import run_grid
from parlai.core.loader import load_teacher_module
from parlai.core.params import ParlaiParser

import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()

"""
Launch a sweep on the cluster to count the data.

Try not to use this very often, the cluster typically shouldn't be used for CPU heavy jobs.
"""


SWEEP_NAME = "count_jobs"
NUM_JOBS = 80
HOURS = 1


def chunks(lst, n):
    for i in range(0, len(lst), n):
        chunk = [str(x) for x in lst[i : i + n]]
        yield chunk


def get_fold_chunks(opt):
    """
    We create an agent from shared here to avoid enqueueing requests
    """
    task_module = load_teacher_module(opt["task"])
    teacher = task_module(opt)  # switch to your teacher here
    fold_chunks = teacher.get_fold_chunks(opt)
    N = len(fold_chunks) // NUM_JOBS + 1
    chunked_chunks = [",".join(chunk) for chunk in chunks(fold_chunks, n=N)]

    return chunked_chunks


def get_grid(teacher, dt):
    pp = ParlaiParser()
    pp.set_params(task=teacher)
    opt = pp.parse_args()
    opt["datatype"] = dt
    opt["no_auto_enqueues"] = True
    fold_chunks = get_fold_chunks(opt)

    grid = {
        "-t": [teacher],
        "--datatype": [dt],
        "--chunks": fold_chunks,
    }

    return grid


if __name__ == "__main__":
    pp = ParlaiParser()
    opt = pp.parse_args()
    TEACHER = opt["task"]

    for dt in ["train:stream", "valid:stream"]:
        grid = get_grid(TEACHER, dt)
        run_grid(
            grid,
            {},
            SWEEP_NAME,
            gpus=0,
            prefix=f"python -u parlai_diplomacy/utils/count_examples/count_examples_single.py",
            cpus=10,
            partition="learnfair",
            jobtime="{}:00:00".format(HOURS),
            hashname=True,
            create_model_file=False,
            include_job_id=True,
            requeue=True,
            copy_env=False,
        )
