#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Last Run: July 7, 2020
Simple script that adds meta data (Turn to phase) and filters non-standard variants to
the pre-processed redacted_messages. Requires variant_ID field.
"""
import sqlite3
import json
import os
import argparse
from tqdm import tqdm
from glob import glob

STANDARD_GAME = "1"


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[str(col[0])] = str(row[idx])
    return d


# Taken from https://github.com/diplomacy/diplomacy/blob/master/diplomacy/integration/webdiplomacy_net/game.py#L20-L27
def turn_to_phase(turn, phase):
    """ Converts a turn and phase to a short phase name e.g. turn 1 - phase 'Retreats' to 'F1901R' """
    year = 1901 + turn // 2

    if phase == "Unknown" and turn == 0:
        return f"S{year}P"

    if phase == "Unknown" and turn != 0:
        return f"S{year}F"

    season = "S" if turn % 2 == 0 else "F"
    if phase == "Builds":
        season = "W"
    phase_type = {"Diplomacy": "M", "Retreats": "R", "Builds": "A"}[phase]
    return season + str(year) + phase_type


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        default="/checkpoint/fairdiplomacy/processed_chat_jsons/"
        "variantID/redacted_messages_runthree*.json",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/checkpoint/fairdiplomacy/processed_chat_jsons/game_phases/",
    )
    args = parser.parse_args()

    tot = len(glob(args.datapath))
    for i, fle in enumerate(glob(args.datapath)):
        print(f"[ Processing data from path {i} / {tot}: {fle} ... ]")
        with open(fle, "r") as f:
            json_data = json.load(f)

        filtered_json = []
        print("Filtering json...")
        for json_entry in tqdm(json_data[2]["data"]):
            if json_entry["variantID"] == STANDARD_GAME:
                game_phase = turn_to_phase(int(json_entry["turn"]), json_entry["phase"])
                json_entry["game_phase"] = game_phase
                filtered_json.append(json_entry)

        print(f"New json size {len(filtered_json)}")
        json_data[2]["data"] = filtered_json
        save_file = os.path.join(args.save_dir, os.path.basename(fle))
        with open(save_file, "w") as f:
            json.dump(json_data, f)

        print(f"Updated json saved to {save_file}")

    print("Done!")
