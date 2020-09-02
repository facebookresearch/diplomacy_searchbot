#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils for loading Diplomacy games.
"""
import parlai.utils.logging as logging
from fairdiplomacy.game import Game
import pydipcc
from parlai_diplomacy.utils.datapath_constants import GAME_JSONS_PATH

from glob import glob
import gzip
import json
import re
import time
from tqdm import tqdm


def load_json(path):
    """
    Simple utility for loading a JSON from a path
    """
    with open(path, "r") as f:
        data = json.load(f)

    return data


def load_from_gz(fle):
    start_time = time.time()
    with gzip.open(fle, "rb") as f:
        dct = json.load(f)
    tot_time = time.time() - start_time
    logging.log(f"Time to load file: {round(tot_time), 2} seconds", level=logging.SPAM)

    return dct


def load_viz_to_dip_format(path):
    """
    Load a game in viz format to diplomacy.Game format
    """
    logging.info(f"Loading viz format data from path: {path}")
    loaded_path = load_json(path)
    game = Game.from_saved_game_format(loaded_path)
    game_json = game.to_saved_game_format()

    return game_json


def load_viz_to_dipcc_format(path):
    """
    Load a game in viz format to pydipCC format
    """
    logging.info(f"Loading viz format data from path: {path}")
    loaded_path = load_json(path)
    game = pydipcc.Game.from_json(json.dumps(loaded_path))  # dipCC game object
    game_json = json.loads(game.to_json())  # get dict version

    return game_json


def load_single_sql_game(path, data_format="dipcc", debug=False):
    """
    Load a single game in SQL format

    :param path: path to Game JSON
    :parm data_format: dipcc or dip -- format to load the data in

    :return: game dict
    """
    game = load_json(path)
    if data_format == "dipcc":
        # convert to dipcc format; this gets rid of some extraneous state fields
        # like "influence"
        try:
            game_obj = pydipcc.Game.from_json(json.dumps(game))  # dipCC game object
            game = json.loads(game_obj.to_json())  # get dict version
        except RuntimeError as e:
            if debug:
                logging.warn(f"Path {path} encountered error: \n{e}")
            return None

    return game


def load_sql_format(dir_path=GAME_JSONS_PATH, game_id_lst=None, debug=False, data_format="dipcc"):
    """
    Load all games in a directory

    Returns a dict of games organized by Game ID.
    """
    all_fles = glob(dir_path)
    if debug:
        all_fles = glob(dir_path)[:5]  # only load 5 games for debugging

    def get_game_id(fle_path):
        return int(re.search("game_(.*).json", fle_path, re.IGNORECASE).group(1))

    game_id_to_fle = {get_game_id(fle_path): fle_path for fle_path in all_fles}
    if game_id_lst is not None:
        logging.info(f"Selecting games with IDs: {game_id_lst}")
        logging.info
        fles = []
        for game_id in game_id_lst:
            if game_id in game_id_to_fle:
                fles.append(game_id_to_fle[game_id])
        else:
            logging.warn(f"Path folder {dir_path} is missing game ID {game_id}; not adding")
        all_fles = fles

    logging.info(f"Loading game data from path {dir_path}, {len(all_fles)} games in total")
    all_games = {}

    missing_orders = 0
    for i, fle in enumerate(tqdm(all_fles)):
        game_id = get_game_id(fle)
        game = load_single_sql_game(fle, data_format=data_format)
        if game is not None:
            # we return None if games are missing phases
            all_games[game_id] = game
        else:
            missing_orders += 1

    logging.warn(f"{missing_orders}/{len(all_fles)} games were excluded due to missing orders.")

    return all_games


def organize_game_by_phase(game_json):
    """
    Organize a single game JSON by phase.
    """
    new_data = {}
    for phase in game_json["phases"]:
        new_data[phase["name"]] = phase

    return new_data


def organize_game_dict_by_phase(game_dict):
    """
    Organize entire game dict by phase.

    Game dict is of the format game ID --> game
    """
    new_data = {}
    for game_id, game in game_dict.items():
        new_data[game_id] = organize_game_by_phase(game)

    return new_data
