#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Script that computes exact match accuracy per orderable location for comparing fairdiplomacy dipnet agent
and parlai diplomacy agent
"""
import json
import torch
from fairdiplomacy.models.dipnet.load_model import load_dipnet_model
from fairdiplomacy.data.dataset import Dataset, DataFields
from fairdiplomacy.models.dipnet.order_vocabulary import *
from fairdiplomacy.game import Game
from fairdiplomacy.data.dataset import ORDER_VOCABULARY_TO_IDX
from fairdiplomacy.models.consts import POWERS
import os
from parlai_diplomacy.tasks.language_diplomacy.utils import COUNTRY_ID_TO_POWER
import joblib
import argparse
import logging
from tqdm import tqdm
from functools import reduce

EOS_IDX = -1


def dipnet_split_accuracy(net, val_set, batch_size):
    """
    Computes exact match accuracy for dipnet model on validation set.
    :param net: dipnet model
    :param val_set: dataset object
    :param batch_size:
    :return: total_orders, total_correct, acc
    """
    net_device = next(net.parameters()).device
    total_orders = 0
    total_correct = 0

    with torch.no_grad():
        net.eval()

        for batch_idxs in tqdm(torch.arange(len(val_set)).split(batch_size)):
            batch = val_set[batch_idxs]
            batch = DataFields({k: v.to(net_device) for k, v in batch.items()})

            y_actions = batch["y_actions"]
            if y_actions.shape[0] == 0:
                print("Got an empty validation batch! y_actions.shape={}".format(y_actions.shape))
                continue

            order_idxs, sampled_idxs, logits, final_sos = net(
                **{k: v for k, v in batch.items() if k.startswith("x_")}, temperature=1e-9
            )

            mask = y_actions != EOS_IDX
            num_orders = mask.sum().float().item()
            y_actions = y_actions[: (sampled_idxs.shape[0]), : (sampled_idxs.shape[1])].to(
                sampled_idxs.device
            )
            num_correct = (y_actions[mask] == sampled_idxs[mask]).float().sum().item()

            total_orders += num_orders
            total_correct += num_correct
    return total_orders, total_correct, total_correct / total_orders


def parlai_split_accuracy(game_id, eval_game_dict):
    """
    Produces the dictionary with true orders and valid locations
    :param game_id: game id
    :param eval_game_dict:
    :return: game_id, game_order_dict
    """
    json_path = os.path.join(args.json_dir, f"game_{game_id}.json")
    try:
        with open(json_path) as f:
            j = json.load(f)
        game = Game.from_saved_game_format(j)
    except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
        print(f"Error while loading game at {json_path}: {e}")
        return None

    total_phases = len(game.state_history)
    # print(f"Encoding {game.game_id} with {total_phases} phases")

    num_total_orders = 0
    num_total_matches = 0
    global_missing_orders = []
    game_dict = dict()
    assert len(eval_game_dict) - 1 == total_phases
    for idx in range(total_phases):
        phase_name = str(list(game.state_history.keys())[idx])
        tmp_game = Game.clone_from(game, up_to_phase=phase_name)
        # all_possible_orders = tmp_game.get_all_possible_orders()
        all_orderable_locations = tmp_game.get_orderable_locations()
        eval_phase_dict = eval_game_dict[phase_name]
        phase_dict = dict()

        # TODO: BUILDS/DISBAND? are an issue
        # TODO: Filter out duplicate orders for same location in eval
        for (id, power) in COUNTRY_ID_TO_POWER.items():
            eval_power_orders = eval_phase_dict[str(id)]["predicted"].split(";")

            missing_orders = [
                order
                for order in game.order_history[phase_name].get(power, [])
                if order not in ORDER_VOCABULARY_TO_IDX
            ]

            global_missing_orders.extend(missing_orders)
            power_orders_set = set(
                [
                    ORDER_VOCABULARY_TO_IDX[order]
                    for order in game.order_history[phase_name].get(power, [])
                    if order in ORDER_VOCABULARY_TO_IDX
                ]
            )

            eval_power_order_set = set(
                [
                    ORDER_VOCABULARY_TO_IDX[order.strip()]
                    for order in eval_power_orders
                    if order.strip() in ORDER_VOCABULARY_TO_IDX
                ]
            )

            num_matches = len(power_orders_set & eval_power_order_set)
            num_orders = len(power_orders_set)

            power_orderable_locs = all_orderable_locations[power]
            phase_dict[power] = {
                "gt_orders": power_orders_set,
                "eval_orders": eval_power_order_set,
                "orderable_locs": power_orderable_locs,
                "num_matches": num_matches,
                "num_orders": num_orders,
            }

            num_total_matches += num_matches
            num_total_orders += num_orders
        game_dict[phase_name] = phase_dict

    return [game_id], [game_dict], num_total_matches, num_total_orders, global_missing_orders


def compute_dipnet_accuracy(p_args):
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

    print("Generating validation jsons...")

    num_orders, num_correct, acc = dipnet_split_accuracy(model, val_set, args.batch_size)
    print(f"num_orders: {num_orders}, num_correct: {num_correct}, accuracy: {acc}")


def compute_parlai_accuracy(args):
    """
    Computes parlai diplomacy model exact match accuracy from json dictionary
    :param args:
    :return:
    """
    # Read evaluation json.
    with open(args.eval_file) as f:
        eval_dict = json.load(f)

    game_ids = list(eval_dict.keys())

    def _combine(a, b):
        return tuple([a_ + b_ for a_, b_ in zip(a, b)])

    output = reduce(
        _combine,
        [
            el
            for el in joblib.Parallel(n_jobs=args.num_jobs)(
                joblib.delayed(parlai_split_accuracy)(game_id, eval_dict[game_id])
                for game_id in game_ids
            )
            if el is not None
        ],
    )

    num_orders = output[3]
    num_correct = output[2]
    acc = output[2] / output[3]

    print(f"Missing {len(output[4])} orders")
    print(
        f"Parlai accuracy: num_orders: {num_orders}, num_correct: {num_correct}, accuracy: {acc}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dipnet", action="store_true",
    )
    parser.add_argument(
        "--parlai", action="store_true",
    )
    parser.add_argument(
        "--json_dir", type=str, default="/checkpoint/fairdiplomacy/processed_orders_jsons/",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="/checkpoint/fairdiplomacy/validation_report/valid_prediction_json/valid_set_prediction.json",
    )
    parser.add_argument(
        "--num_jobs", type=int, default=20,
    )
    parser.add_argument(
        "--checkpoint",
        default="/checkpoint/alerer/fairdiplomacy/sl_fbdata_all/checkpoint.pth.best",
    )
    parser.add_argument(
        "--data-cache",
        default="/checkpoint/alerer/fairdiplomacy/facebook_notext/data_cache_fb_minrating0.5.pt",
    )
    args = parser.parse_args()

    if args.parlai:
        compute_parlai_accuracy(args)

    if args.dipnet:
        compute_dipnet_accuracy(args)
