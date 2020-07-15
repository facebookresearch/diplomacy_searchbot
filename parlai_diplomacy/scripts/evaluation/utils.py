#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from fairdiplomacy.game import Game
import os


def load_game(json_dir, game_id):
    json_path = os.path.join(json_dir, f"game_{game_id}.json")
    try:
        with open(json_path) as f:
            j = json.load(f)
        game = Game.from_saved_game_format(j)
        return game
    except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
        print(f"Error while loading game at {json_path}: {e}")
        return None