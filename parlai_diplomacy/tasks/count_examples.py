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

from parlai_diplomacy.tasks.dialogue.agents import utls, BaseDialogueChunkTeacher
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
                            variant == "full_press"
                        ):  # press_dialogue: Train: 10463131, Valid: 557029
                            for power, value in phase.items():
                                from_msgs = [
                                    s
                                    for s in phase[power]["message"].split("\n")
                                    if s.startswith(COUNTRY_ID_TO_POWER[int(power)].capitalize())
                                ]
                                num_msgs = len(from_msgs) + (1 if variant == "full_press" else 0)
                                # Add 1 for orders
                                tot += num_msgs
                                fle_total += num_msgs
                        elif variant == "press_dialogue":
                            # press_dialogue_new: Train: 6120613, Valid: 327647
                            for power, value in phase.items():
                                if value["data_status"] != "Game_Phase_Msg":
                                    break

                                num_msgs = 0

                                cur_power = utls.COUNTRY_ID_TO_POWER[int(power)].capitalize()
                                messages = value["message"]
                                message_history = value["message_history"].replace(messages, "")

                                all_msgs = messages.split("\n")
                                message_history_list = message_history.split("\n")

                                all_msgs = BaseDialogueChunkTeacher.add_silence_messages(
                                    cur_power, all_msgs, message_history_list
                                )

                                cur_power_consec = False
                                for msg in all_msgs:
                                    if msg.startswith(cur_power):
                                        if not cur_power_consec:
                                            cur_power_consec = True
                                            num_msgs += 1
                                    else:
                                        cur_power_consec = False

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
