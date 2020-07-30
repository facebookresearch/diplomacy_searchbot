#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Simple script for iterating through the chunk data to count examples.

Committing here as an example for iterating/performing analysis on this data
"""
import parlai_diplomacy.tasks.common_task_utils as utls
from parlai_diplomacy.tasks.dialogue_and_state_stream.agents import TRAIN_VAL_SPLIT
from parlai.utils import logging

from glob import glob
import json
import os


if __name__ == "__main__":
    file_lst = sorted(glob(utls.CHUNK_ORDER_PATH))
    train = file_lst[:TRAIN_VAL_SPLIT]
    valid = file_lst[TRAIN_VAL_SPLIT:]
    for dt, lst in [("train", train), ("valid", valid)]:
        by_fle = {}
        tot = 0
        for i, fle in enumerate(lst):
            fle_total = 0
            with open(fle, "r") as f:
                data = json.load(f)
                for _, game in data.items():
                    for _, phase in game.items():
                        tot += len(phase)
                        fle_total += len(phase)
                if i > 0 and i % 10 == 0:
                    complete = round((i / len(lst)) * 100, 2)
                    logging.info(f"({complete}%) Loaded {tot} {dt} examples so far...")
            by_fle[fle] = fle_total

        print("\n\n\n\n\n")
        print("=" * 50)
        print(f"FINISHED {dt}: loaded {tot} total examples.")
        print("By file: ")
        for fle, total in by_fle.items():
            print(f"{os.path.basename(fle)}:\t{total}")
