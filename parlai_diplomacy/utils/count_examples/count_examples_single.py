#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Count data for a set of chunks. DO NOT USE: called in `count_examples_sweep.py`

Example usage:

```
pyhon parlai_diplomacy/tasks/count_examples.py -t message_order_chunk -dt valid:stream --chunks 0,1,2
```
"""
from parlai.utils import logging
from parlai.core.params import ParlaiParser
from parlai.core.loader import load_teacher_module

import subprocess
import os

import parlai_diplomacy.utils.loading as load

load.register_all_agents()
load.register_all_tasks()


def count_single_chunk(opt, chunk_idx):
    task_module = load_teacher_module(opt["task"])
    teacher = task_module(opt)
    exs = teacher.load_from_chunk(chunk_idx)

    del teacher

    return len(exs)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def bash(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, _ = process.communicate()
    output = str(output)
    output = output[:-3]
    output = output.lstrip("b").strip("'").strip('"')
    return output


def get_file_path(opt):
    user = os.environ["USER"]
    date = bash('date +"%Y%m%d"')
    task = opt["task"].replace(":", "_")
    dt = opt["datatype"].split(":")[0]
    path = os.path.join(f"/checkpoint/{user}/diplomacy_data_cnts/{date}/{task}/{dt}/")
    ensure_dir(path)
    return path


if __name__ == "__main__":
    pp = ParlaiParser()
    pp.add_argument("--chunks", type=str, required=True)
    pp.add_argument("--write-folder", type=str, default=None)
    opt = pp.parse_args()
    opt["no_auto_enqueues"] = True
    if opt["write_folder"] is None:
        opt["write_folder"] = get_file_path(opt)
        logging.info(f"Creating folder {opt['write_folder']} for example counts")

    chunk_idxs = [int(x) for x in opt["chunks"].split(",")]
    for chunk_idx in chunk_idxs:
        path = os.path.join(opt["write_folder"], f"{chunk_idx}.txt")
        if os.path.isfile(path):
            # already finished this
            continue
        count = count_single_chunk(opt, chunk_idx)
        folder_write = path
        with open(folder_write, "w") as f:
            f.write(str(count))
