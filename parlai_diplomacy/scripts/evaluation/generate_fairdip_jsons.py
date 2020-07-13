#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script that generates fairdiplomacy evaluation jsons to be used in compute_accuracy.py
"""
import json
import torch
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.data.dataset import Dataset, DataFields
from fairdiplomacy.models.dipnet.order_vocabulary import *
from fairdiplomacy.game import Game
from fairdiplomacy.data.dataset import ORDER_VOCABULARY_TO_IDX, ORDER_VOCABULARY
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.agents.dipnet_agent import encode_inputs
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model, new_model
import os
from parlai_diplomacy.tasks.language_diplomacy.utils import COUNTRY_ID_TO_POWER
import joblib
import argparse
import logging
from tqdm import tqdm, trange
from functools import reduce
from parlai_diplomacy.scripts.evaluation.utils import load_game


EOS_IDX = -1


def generate_jsons(net, args, game_ids):
    """

    :param net:
    :param args:
    :param game_ids:
    :return:
    """

    game_dict = dict()
    for game_id in tqdm(game_ids, desc="Games"):
        game = load_game(args.json_dir, game_id)
        if game is None:
            continue

        phase_dict = dict()
        total_phases = len(game.state_history)
        for idx in range(total_phases):
            phase_name = str(list(game.state_history.keys())[idx])
            tmp_game = Game.clone_from(game, up_to_phase=phase_name)

            with torch.no_grad():
                net.eval()

                inputs = encode_inputs(tmp_game)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

                order_idxs, _, _, _ = net(**inputs, temperature=1e-9)

                orders = order_idxs.squeeze(0).cpu().tolist()

                country_dict = {
                    power: ";".join(
                        [ORDER_VOCABULARY[order] for order in orders[id] if order != EOS_IDX]
                    )
                    for id, power in enumerate(POWERS)
                }

            phase_dict[phase_name] = country_dict
        game_dict[game_id] = phase_dict

    save_file = os.path.join(args.save_dir, "fairdip_valid_set_prediction.json")
    with open(save_file, "w") as f:
        json.dump(game_dict, f)

    print(f"Validation predict json saved to {save_file}")


def generate_dipnet_jsons(p_args):
    """
    Computes fairdiplomacy dipnet exact match accuracy using greedy decoding
    :param p_args:
    :return:
    """
    print("Loading model")
    model = load_dipnet_model(p_args.checkpoint, map_location="cuda", eval=True)
    args = torch.load(p_args.checkpoint)["args"]

    print("Loading dataset")
    _, val_set = torch.load(args.data_cache)

    # Read evaluation json.
    with open(p_args.eval_file) as f:
        eval_dict = json.load(f)

    game_ids = val_set.game_ids  # list(eval_dict.keys())

    print("Generating validation jsons...")
    generate_jsons(model, p_args, game_ids)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir", type=str, default="/checkpoint/fairdiplomacy/processed_orders_jsons/",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="/checkpoint/fairdiplomacy/validation_report/valid_prediction_json/valid_set_prediction.json",
    )
    parser.add_argument(
        "--save_dir", type=str, default="/private/home/apjacob",
    )
    parser.add_argument(
        "--num_jobs", type=int, default=20,
    )
    parser.add_argument(
        "--checkpoint",
        default="/checkpoint/alerer/fairdiplomacy/sl_fbdata_all/checkpoint.pth.best",
    )
    args = parser.parse_args()

    generate_dipnet_jsons(args)
