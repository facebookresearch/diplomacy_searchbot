#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import parlai_diplomacy.tasks.language_diplomacy.utils as utils
from parlai_diplomacy.tasks.language_diplomacy.agents import TRAIN_SPLIT

from collections import defaultdict

"""
Simple script for calculating the number of examples/episodes in
the diplomacy data based on the script.

This is used because for the streaming teacher, we need to know in
advance how many examples there are.
"""


if __name__ == "__main__":
    iterator = utils.DataIterator()
    # Run through all turns to get a list of conversations
    convs = []
    print(f"Total dataset Diplomacy turns: {len(iterator)}")
    tot_turns = 0
    train_cnts = defaultdict(int)
    valid_cnts = defaultdict(int)

    train_eps = 0
    valid_eps = 0

    for i, turn in enumerate(iterator):
        for conv in turn:
            length = len(conv)
            if i < TRAIN_SPLIT:
                train_cnts[length] += 1
                train_eps += 1
            else:
                # valid
                valid_cnts[length] += 1
                valid_eps += 1

    for i in range(100):
        tot_train = sum(train_cnts[j] for j in train_cnts.keys() if j > i)
        tot_valid = sum(valid_cnts[j] for j in valid_cnts.keys() if j > i)

        tot_train_exs = 0
        for k, v in train_cnts.items():
            if k > i:
                tot_train_exs += k * v

        tot_valid_exs = 0
        for k, v in valid_cnts.items():
            if k > i:
                tot_valid_exs += k * v

        print(f"Num train episodes, exs with length >= {i + 1}: {tot_train}, {tot_train_exs}")
        print("----------")
        print(f"Num valid episodes, exs with length >= {i + 1}: {tot_valid}, {tot_valid_exs}")
        print("=========\n")

        if tot_train == 0 and tot_valid == 0:
            break
