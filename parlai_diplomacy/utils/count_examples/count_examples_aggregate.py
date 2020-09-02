#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
To be used after running `count_examples_sweep.py` script in the same folder.


Example usage:

```
python /parlai_diplomacy/utils/count_examples/count_examples_aggregate.py -t state_order_chunk
```
"""
from parlai.utils import logging
from parlai.core.params import ParlaiParser

import parlai_diplomacy.utils.loading as load
from parlai_diplomacy.utils.count_examples.count_examples_single import get_file_path

import os

load.register_all_agents()
load.register_all_tasks()


def all_chunk_total(path, print_per_chunk=False):
    total = 0
    logging.info(f"Loading all files from path: {path}")
    all_chunks = os.listdir(path)
    for chunk in all_chunks:
        with open(os.path.join(path, chunk), "r") as f:
            line = int(f.readlines()[0])
            if print_per_chunk:
                logging.info(f"Chunk: {chunk.split('.txt')[0]} has {line}")
            total += line

    return total, len(all_chunks)


if __name__ == "__main__":
    pp = ParlaiParser()
    pp.add_argument("--write-folder", type=str, default=None)
    pp.add_argument("--print-per-chunk", type="bool", default=False)
    opt = pp.parse_args()

    if opt["write_folder"] is None:
        opt["write_folder"] = get_file_path(opt)

    tot, num = all_chunk_total(opt["write_folder"], print_per_chunk=opt["print_per_chunk"])
    logging.success(f"Total examples ({opt['datatype']}): {tot} from {num} chunks")
