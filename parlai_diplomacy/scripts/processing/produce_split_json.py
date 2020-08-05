#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script that generates jsons which have the same train/valid splits that fairdiplomacy uses
python produce_split_json.py --data_cache=[DATA_CACHE] --json_dir=[PROCESSED_GAME_JSONS] --save_dir=[SAVE_DIR]
"""
import argparse
import json
import os

import torch
from tqdm import tqdm

from fairdiplomacy.game import Game
from fairdiplomacy.models.consts import POWERS


def get_game(game_id, data_dir):
    """
    Creates game object from game_id
    :param game_id:
    :param data_dir:
    :return:
    """
    game_path = os.path.join(f"{data_dir}", f"game_{game_id}.json")

    with open(game_path) as f:
        j = json.load(f)
    game = Game.from_saved_game_format(j)

    return game


def save_split_json(dataset, save_file, data_dir):
    """
    Saves splits in json format
    :param dataset:
    :param save_file:
    :param data_dir:
    :return:
    """
    game_idxs = dataset.game_idxs.data.tolist()
    phase_ids = dataset.phase_idxs.data.tolist()
    power_ids = dataset.power_idxs.data.tolist()

    cur_game = None
    cur_game_id = None
    game_dict = {}
    for i in tqdm(range(len(game_idxs))):

        game_idx = game_idxs[i]
        game_id = dataset.game_ids[game_idx]
        game_dict.setdefault(game_id, {})

        # To avoid loading games multiple times
        if game_id != cur_game_id:
            cur_game = get_game(game_id, data_dir)
            cur_game_id = game_id

        phase_id = phase_ids[i]
        phase_name = cur_game.get_phase_name(phase_id)

        power_id = power_ids[i]
        power_name = POWERS[power_id]

        game_dict[game_id].setdefault(phase_name, [])
        game_dict[game_id][phase_name].append(power_name)

    with open(save_file, "w") as f:
        json.dump(game_dict, f)

    print(f"Split saved to {save_file}")


def create_split_jsons(args):
    print("Loading dataset")
    train_set, val_set = torch.load(args.data_cache)

    file_name = os.path.basename(args.data_cache)

    print("Creating valid split..")
    save_split_json(
        dataset=val_set,
        save_file=os.path.join(args.save_dir, f"{file_name}_valid.json"),
        data_dir=args.json_dir,
    )

    print("Creating train split..")
    save_split_json(
        dataset=train_set,
        save_file=os.path.join(args.save_dir, f"{file_name}_train.json"),
        data_dir=args.json_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir", type=str, default="/checkpoint/fairdiplomacy/processed_orders_jsons/",
    )
    parser.add_argument(
        "--data_cache",
        type=str,
        default="/checkpoint/apjacob/fairdiplomacy/no_press/data_cache/data_cache_fb_minrating0.5.pt",
    )
    parser.add_argument(
        "--save_dir", type=str, default="/tmp/",
    )

    args = parser.parse_args()
    create_split_jsons(args)
