#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Count data for chunk teachers.

Example usage:

```
python parlai_diplomacy/tasks/count_examples.py -t message_order_chunk
```
"""
from parlai.utils import logging
from parlai.core.agents import create_agent_from_shared
from parlai.core.params import ParlaiParser
from parlai.core.loader import load_teacher_module
import parlai_diplomacy.utils.datapath_constants as constants

import parlai_diplomacy.utils.loading as load

from copy import deepcopy
from collections import defaultdict
import multiprocessing
import json
from tqdm import tqdm

load.register_all_agents()
load.register_all_tasks()

NUM_PROCESSES = 20  # change this depending on your CPU/mem needs


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_fold_chunks(opt):
    """
    We create an agent from shared here to avoid enqueueing requests
    """
    task_module = load_teacher_module(opt["task"])
    teacher = task_module(opt)  # switch to your teacher here
    fold_chunks = teacher.get_fold_chunks(opt)

    del teacher

    return fold_chunks


def count_single_chunk(opt, chunk_idx):
    task_module = load_teacher_module(opt["task"])
    teacher = task_module(opt)
    exs = teacher.load_from_chunk(chunk_idx)

    del teacher

    return chunk_idx, len(exs)


def count_single_chunk_wrapper(args):
    return count_single_chunk(*args)


if __name__ == "__main__":
    pp = ParlaiParser()
    opt = pp.parse_args()
    total_dct = defaultdict(int)
    chunk_idx_to_num_examples = defaultdict(int)
    for dt in ["train:stream", "valid:stream"]:
        opt["datatype"] = dt
        opt["no_auto_enqueues"] = True
        total = 0
        fold_chunks = get_fold_chunks(opt)
        for i, chunk_lst in tqdm(enumerate(chunks(fold_chunks, n=NUM_PROCESSES))):
            args = [(deepcopy(opt), chunk_num) for chunk_num in chunk_lst]
            logging.info(f"Creating pool: {i} with chunks: {chunk_lst}")
            pool = multiprocessing.Pool(processes=NUM_PROCESSES)
            ret = pool.map(count_single_chunk_wrapper, args)
            logging.info(f"Return value for pool: {i} is {ret}")
            pool.close()
            pool.join()
            chunk_idx_to_num_examples.update(
                {chunk_idx_tmp: num_example_tmp for chunk_idx_tmp, num_example_tmp in ret}
            )
            total += sum([num_example_tmp for _, num_example_tmp in ret])
            logging.warn(f"Current subtotal: {total}")

        logging.success(f"Final value for datatype: {dt} -- {total} examples")
        total_dct[dt] = total

    with open(constants.CHUNK_EXAMPLE_COUNT_PATH, "w") as fh:
        json.dump(chunk_idx_to_num_examples, fh)
        logging.success(
            f"Successfully saved example_counts to {constants.CHUNK_EXAMPLE_COUNT_PATH}"
        )

    for dt, cnt in total_dct.items():
        logging.success(f"Final count for {dt}: {cnt}")
