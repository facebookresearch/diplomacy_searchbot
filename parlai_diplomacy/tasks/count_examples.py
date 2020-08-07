#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Simple script for iterating through the chunk data to count examples.

Committing here as an example for iterating/performing analysis on this data
"press_dialogue" -> [messages, state] -> [message]
"full_press" -> [messages, state] -> [message or order]
"""
import parlai_diplomacy.tasks.common_task_utils as utls
from parlai_diplomacy.tasks.no_press.stream.agents import TRAIN_VAL_SPLIT
from parlai.utils import logging
from parlai_diplomacy.tasks.common_task_utils import COUNTRY_ID_TO_POWER
from glob import glob
import json
import os
import argparse


def print_count(variant):
    train, valid = get_split_chunk_files()
    for dt, lst in [("train", train), ("valid", valid)]:
        by_fle = {}
        tot = 0
        for i, fle in enumerate(lst):
            fle_total = 0
            with open(fle, "r") as f:
                data = json.load(f)
                for _, game in data.items():
                    for _, phase in game.items():
                        if variant == "default":
                            tot += len(phase)
                            fle_total += len(phase)
                        elif (
                            variant == "press_dialogue" or variant == "full_press"
                        ):  # press_dialogue: Train: 10463131, Valid: 557029
                            for power, value in phase.items():
                                from_msgs = [
                                    s
                                    for s in phase[power]["message"].split("\n")
                                    if s.startswith(COUNTRY_ID_TO_POWER[int(power)].capitalize())
                                ]
                                num_msgs = len(from_msgs) + (
                                    1 if variant == "full_press" else 0
                                )  # Add 1 for orders
                                tot += num_msgs
                                fle_total += num_msgs

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


def get_split_chunk_files():
    file_lst = sorted(glob(utls.CHUNK_ORDER_PATH))
    train = file_lst[:TRAIN_VAL_SPLIT]
    valid = file_lst[TRAIN_VAL_SPLIT:]
    return train, valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count number of examples for chunk teachers")
    parser.add_argument(
        "--variant",
        choices=["default", "press_dialogue", "full_press"],
        help="Stream teacher variant",
    )
    args = parser.parse_args()

    print_count(args.variant)
